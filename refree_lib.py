# ===================== refree_lib.py : all project code, loaded once per Colab session =====================
import os, json, math, time, random, warnings, pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT  = Path("/content/drive/MyDrive")
CACHE = ROOT / "refree_cache"      # aligned faces (npz shards)
CK    = ROOT / "refree_ckpt"       # priors.pt + items_meta.pkl
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = SimpleNamespace(
    SEED=42, IMG=224, FRAMES_PER_VIDEO=3,
    FFPP_ROOT="/content/FaceForensics++", FFPP_COMPRESSION="c23", CELEBDF_ROOT="/content/Celeb-DF-v2",
    DLIB_PREDICTOR="/content/shape_predictor_68_face_landmarks.dat")
S = cfg.IMG
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)

def make_template_68():
    pts = []
    t = np.linspace(0, np.pi, 17)
    pts += list(zip(112 - 80*np.cos(t), 85 + 110*np.sin(t)))
    u = np.linspace(0, np.pi, 5)
    pts += list(zip(np.linspace(48, 98, 5), 62 - 8*np.sin(u)))
    pts += list(zip(np.linspace(126, 176, 5), 62 - 8*np.sin(u)))
    pts += [(112, 75), (112, 90), (112, 105), (112, 118)]
    pts += list(zip(np.linspace(94, 130, 5), [130, 134, 136, 134, 130]))
    ang = np.deg2rad([180, 125, 55, 0, -55, -125])
    for cx in (76, 148):
        pts += list(zip(cx + 17*np.cos(ang), 86 - 7*np.sin(ang)))
    a = np.deg2rad([180, 150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150])
    pts += list(zip(112 + 32*np.cos(a), 165 - 14*np.sin(a)))
    a = np.deg2rad([180, 120, 90, 60, 0, -60, -90, -120])
    pts += list(zip(112 + 20*np.cos(a), 165 - 5*np.sin(a)))
    return np.array(pts, dtype=np.float32)
TEMPLATE_68 = make_template_68()
STABLE_PTS = [36, 39, 42, 45, 30, 48, 54]

# ---- face detection / alignment (only needed for NEW images; needs dlib + the predictor file) ----
_dlib = {}
def detect_landmarks(frame_bgr):
    if "det" not in _dlib:
        import dlib
        _dlib["det"] = dlib.get_frontal_face_detector(); _dlib["sp"] = dlib.shape_predictor(cfg.DLIB_PREDICTOR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _dlib["det"](gray, 1)
    if not faces: return None
    f = max(faces, key=lambda r: r.width() * r.height())
    return np.array([[p.x, p.y] for p in _dlib["sp"](gray, f).parts()], dtype=np.float32)
def detect_landmarks_fast(frame, DET_MAX=640):
    h, w = frame.shape[:2]; s = min(1.0, DET_MAX / max(h, w))
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
    lm = detect_landmarks(small)
    return None if lm is None else (lm / s).astype(np.float32)
def align_face(frame_bgr, lm, size=S):
    M, _ = cv2.estimateAffinePartial2D(lm[STABLE_PTS], TEMPLATE_68[STABLE_PTS], method=cv2.LMEDS)
    crop = cv2.warpAffine(frame_bgr, M, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return crop, (lm @ M[:, :2].T + M[:, 2]).astype(np.float32)

# ===================== priors / autoencoder / fusion =====================

# ---------------- region masks (notebook §1) ----------------
REGION_NAMES = ["left eye", "right eye", "nose", "mouth", "face boundary"]
def region_masks(lm, size=S):
    '''Boolean masks for the facial parts used by the symmetry analysis, prompts and scoring.'''
    lm = np.asarray(lm)
    def poly(idx, grow=0):
        m = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(lm[idx].astype(np.int32)), 1)
        if grow:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
            m = cv2.dilate(m, k)
        return m.astype(bool)
    hull = poly(list(range(0, 27)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    ring = cv2.dilate(hull.astype(np.uint8), k).astype(bool) & ~cv2.erode(hull.astype(np.uint8), k).astype(bool)
    return {"left eye": poly(list(range(42, 48)), 11), "right eye": poly(list(range(36, 42)), 11),
            "nose": poly(list(range(27, 36)), 11), "mouth": poly(list(range(48, 68)), 11),
            "face boundary": ring, "face": hull}

# ---------------- reference-free priors (notebook §3) ----------------
SRM_KERNELS = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2,
]
def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
def srm_residuals(gray):
    return [cv2.filter2D(gray, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT) for k in SRM_KERNELS]
def srm_log_energy(gray, win=13):
    e = sum(cv2.boxFilter(r * r, -1, (win, win), borderType=cv2.BORDER_REFLECT) for r in srm_residuals(gray)) / len(SRM_KERNELS)
    return np.log(e + 1e-3)

_HANN = np.outer(np.hanning(32), np.hanning(32)).astype(np.float32)
_FREQ = np.sqrt(np.fft.fftfreq(32)[:, None] ** 2 + np.fft.fftfreq(32)[None, :] ** 2)
def fft_hf_map(gray, win=32, stride=8, r0=0.25):
    from numpy.lib.stride_tricks import sliding_window_view
    H, W = gray.shape
    gp = np.pad(gray, win // 2, mode="reflect")
    w = sliding_window_view(gp, (win, win))[::stride, ::stride]
    w = (w - w.mean((-1, -2), keepdims=True)) * _HANN
    P = np.abs(np.fft.fft2(w)) ** 2
    ratio = np.log(P[..., _FREQ > r0].sum(-1) / (P.sum((-1, -2)) + 1e-6) + 1e-6).astype(np.float32)
    return cv2.resize(ratio, (W, H), interpolation=cv2.INTER_CUBIC)

def midline_x(lm):
    return float(np.asarray(lm)[[27, 28, 29, 30, 33, 51, 57, 8], 0].mean())
def mirror(x, xm):
    M = np.float32([[-1, 0, 2 * xm], [0, 1, 0]])
    return cv2.warpAffine(x, M, (x.shape[1], x.shape[0]), borderMode=cv2.BORDER_REFLECT)
def symmetry_maps(gray, logE, lm):
    xm = midline_x(lm)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 4)
    d_tex = cv2.GaussianBlur(np.abs(hp - mirror(hp, xm)), (0, 0), 3)
    d_res = cv2.GaussianBlur(np.abs(logE - mirror(logE, xm)), (0, 0), 3)
    return d_tex, d_res

PRIOR_NAMES = ["srm", "fft", "sym_tex", "sym_res", "ae"]
ONE_SIDED = set()   # v2: every prior uses two-sided |deviation| (v1 was {"sym_tex","sym_res","ae"})
def raw_maps(img_bgr, lm):
    g = to_gray(img_bgr); logE = srm_log_energy(g)
    d_tex, d_res = symmetry_maps(g, logE, lm)
    return {"srm": logE, "fft": fft_hf_map(g), "sym_tex": d_tex, "sym_res": d_res}

def radial_spectrum(img_bgr, nb=8):
    g = to_gray(img_bgr); g = (g - g.mean()) * np.outer(np.hanning(S), np.hanning(S))
    P = np.fft.fftshift(np.abs(np.fft.fft2(g)) ** 2)
    yy, xx = np.indices(P.shape); r = np.hypot(yy - S / 2, xx - S / 2) / (S / 2)
    edges = np.linspace(0, 1, nb + 1)
    return np.array([np.log(P[(r >= edges[i]) & (r < edges[i + 1])].mean() + 1e-6) for i in range(nb)], np.float32)

# ---------------- pseudo-reference autoencoder (notebook §4) ----------------
class FaceAE(nn.Module):
    def __init__(self, c=(16, 32, 64), z=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, c[0], 4, 2, 1), nn.ReLU(True), nn.Conv2d(c[0], c[1], 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(c[1], c[2], 4, 2, 1), nn.ReLU(True), nn.Conv2d(c[2], z, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(z, c[2], 1), nn.ReLU(True), nn.ConvTranspose2d(c[2], c[1], 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(c[1], c[0], 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(c[0], 3, 4, 2, 1), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))

def img_to_tensor01(imgs):
    a = np.stack([im[..., ::-1] for im in imgs]).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()

def train_ae(real_imgs, epochs=15, bs=16, lr=2e-3):
    ae = FaceAE().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    X = torch.from_numpy(np.stack([im[..., ::-1] for im in real_imgs])).permute(0, 3, 1, 2).contiguous()   # uint8, saves RAM
    for ep in range(epochs):
        perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            xb = X[perm[i:i + bs]].to(DEVICE).float() / 255.0
            loss = F.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(xb)
        if ep in (0, epochs - 1) or (ep + 1) % 5 == 0:
            print(f"  AE epoch {ep+1:2d}/{epochs}  recon MSE = {tot/len(X):.5f}", flush=True)
    return ae.eval()

@torch.no_grad()
def ae_diff_maps(ae, imgs, bs=32):
    '''|i_hat_r - i_f| (channel-mean, lightly smoothed). Returns (None, diffs) to keep the notebook signature.'''
    out_diff = []
    for i in range(0, len(imgs), bs):
        x = img_to_tensor01(imgs[i:i + bs]).to(DEVICE)
        d = (ae(x) - x).abs().mean(1).cpu().numpy()
        out_diff += [cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2) for m in d]
    return None, out_diff

# ---------------- population statistics + fusion (notebook §5) ----------------
class ForensicPriors:
    Z_CAP = 4.0
    WEIGHTS = {"srm": 0.30, "fft": 0.20, "sym_tex": 0.20, "sym_res": 0.15, "ae": 0.15}

    def fit(self, real_imgs, lms=None, ae_epochs=15):
        lms = lms if lms is not None else [TEMPLATE_68] * len(real_imgs)
        print(f"fitting priors on {len(real_imgs)} real faces ...", flush=True)
        self.ae = train_ae(real_imgs, epochs=ae_epochs)
        s1 = {k: np.zeros((S, S), np.float64) for k in PRIOR_NAMES}; s2 = {k: np.zeros((S, S), np.float64) for k in PRIOR_NAMES}
        n = 0
        for i in range(0, len(real_imgs), 32):          # running sums instead of stacking every map (RAM-safe)
            ims, ls = real_imgs[i:i + 32], lms[i:i + 32]
            _, ae_d = ae_diff_maps(self.ae, ims)
            for im, lm, ad in zip(ims, ls, ae_d):
                m = raw_maps(im, lm); m["ae"] = ad
                for k in PRIOR_NAMES:
                    x = m[k].astype(np.float64); s1[k] += x; s2[k] += x * x
                n += 1
            if (i // 32) % 20 == 0: print(f"  stats {n}/{len(real_imgs)}", flush=True)
        self.mu, self.sd = {}, {}
        for k in PRIOR_NAMES:
            mu = s1[k] / n; sd = np.sqrt(np.maximum(s2[k] / n - mu * mu, 0))
            mu = cv2.GaussianBlur(mu.astype(np.float32), (0, 0), 3)
            sd = cv2.GaussianBlur(sd.astype(np.float32), (0, 0), 3)
            self.mu[k], self.sd[k] = mu, np.maximum(sd, 0.25 * np.median(sd))
        return self

    def z_maps(self, imgs, lms=None):
        lms = lms if lms is not None else [TEMPLATE_68] * len(imgs)
        _, ae_d = ae_diff_maps(self.ae, imgs)
        out = []
        for im, lm, ad in zip(imgs, lms, ae_d):
            m = raw_maps(im, lm); m["ae"] = ad
            z = {}
            for k in PRIOR_NAMES:
                d = m[k] - self.mu[k]
                z[k] = (np.maximum(d, 0) if k in ONE_SIDED else np.abs(d)) / self.sd[k]
            out.append(z)
        return out

    def fuse(self, z):
        acc = sum(self.WEIGHTS[k] * np.clip(z[k] / self.Z_CAP, 0, 1) for k in PRIOR_NAMES)
        return cv2.GaussianBlur(acc.astype(np.float32), (0, 0), 4)

# ===================== analyzer / prompts =====================

# ---------------- prompt generation (notebook §6) ----------------
ARTIFACT_TEXT = {
    "srm": "noise-level inconsistency", "fft": "abnormal high-frequency spectrum (blur / resampling)",
    "sym_tex": "atypical texture symmetry across the facial midline", "sym_res": "atypical left-right noise pattern",
    "ae": "deviation from the reconstructed natural face"}
REAL_HDR = "A real face with natural texture and consistent symmetry."
FAKE_HDR = "A fake face with manipulated regions and blending artifacts."

class ForensicAnalyzer:
    def __init__(self, priors):
        self.pr = priors
        self.reg_thr = {r: 0.2 for r in REGION_NAMES}
        self.prior_thr = {(k, r): 0.25 for k in PRIOR_NAMES for r in REGION_NAMES}

    def _region_stats(self, z, fused, lm):
        masks = region_masks(lm)
        zc = {k: np.clip(z[k] / self.pr.Z_CAP, 0, 1) for k in PRIOR_NAMES}
        reg = {r: float(fused[masks[r]].mean()) for r in REGION_NAMES}
        pri = {(k, r): float(zc[k][masks[r]].mean()) for k in PRIOR_NAMES for r in REGION_NAMES}
        return reg, pri

    def analyze_batch(self, imgs, lms=None):
        lms = lms if lms is not None else [TEMPLATE_68] * len(imgs)
        Z = self.pr.z_maps(imgs, lms)
        res = []
        for im, lm, z in zip(imgs, lms, Z):
            fused = self.pr.fuse(z)
            reg, pri = self._region_stats(z, fused, lm)
            res.append(dict(fused=fused, reg=reg, pri=pri, spec=radial_spectrum(im)))
        return res

    def calibrate(self, real_val_imgs, lms=None, q=95):
        lms = lms if lms is not None else [TEMPLATE_68] * len(real_val_imgs)
        regs, pris = [], []
        for i in range(0, len(real_val_imgs), 32):
            for a in self.analyze_batch(real_val_imgs[i:i + 32], lms[i:i + 32]):
                regs.append(a["reg"]); pris.append(a["pri"])
        for r in REGION_NAMES:
            self.reg_thr[r] = float(np.percentile([x[r] for x in regs], q))
            for k in PRIOR_NAMES:
                self.prior_thr[(k, r)] = float(np.percentile([x[(k, r)] for x in pris], q))
        return self

    def feature_vector(self, a):
        v = [a["pri"][(k, r)] for k in PRIOR_NAMES for r in REGION_NAMES]
        return np.concatenate([v, a["spec"], [a["fused"].mean(), a["fused"].max()]]).astype(np.float32)

    def prompt(self, a, image_id="img"):
        regions = []
        for r in REGION_NAMES:
            ratio = a["reg"][r] / max(self.reg_thr[r], 1e-6)
            if ratio <= 1.0: continue
            arts = [k for k in PRIOR_NAMES if a["pri"][(k, r)] > self.prior_thr[(k, r)]]
            arts = sorted(arts, key=lambda k: -a["pri"][(k, r)] / max(self.prior_thr[(k, r)], 1e-6))[:3]
            if r == "face boundary" and any(k in arts for k in ("srm", "fft")):
                arts = ["blending seam"] + [ARTIFACT_TEXT[k] for k in arts if k not in ("srm", "fft")]
            else:
                arts = [ARTIFACT_TEXT[k] for k in arts]
            sev = "high" if ratio > 2.0 else "medium" if ratio > 1.4 else "low"
            regions.append(dict(region=r, severity=sev, score=round(a["reg"][r], 3), artifacts=arts or ["unspecified local anomaly"],
                                evidence={k: round(a["pri"][(k, r)], 3) for k in PRIOR_NAMES}))
        regions.sort(key=lambda d: -d["score"])
        overall = float(max([a["reg"][r] / max(self.reg_thr[r], 1e-6) for r in REGION_NAMES]))
        if regions:
            names = ", ".join(d["region"] for d in regions[:3])
            reasoning = (f"Heuristic priors flag {names}: " +
                         "; ".join(f"{d['region']} shows {' and '.join(d['artifacts'][:2])}" for d in regions[:3]) + ".")
        else:
            reasoning = "No region exceeds the calibrated real-face thresholds; texture, noise and symmetry look natural."
        return dict(image_id=image_id, heuristic_suspicion=round(min(overall / 3, 1.0), 3), manipulated_regions=regions,
                    global_cues=dict(mean_fused_anomaly=round(float(a["fused"].mean()), 4), peak_fused_anomaly=round(float(a["fused"].max()), 3)),
                    reasoning=reasoning)

def fusion_text(pj):
    if not pj["manipulated_regions"]:
        return "Heuristic findings: none. Texture, noise and symmetry are consistent."
    parts = [f"{d['region']} ({d['severity']}: {', '.join(d['artifacts'][:2])})" for d in pj["manipulated_regions"][:3]]
    return "Heuristic findings: " + "; ".join(parts) + "."

def align_text(pj, label):
    if label == 0: return REAL_HDR
    if not pj["manipulated_regions"]: return FAKE_HDR + " Subtle manipulation, region not localized."
    regs = ", ".join(d["region"] for d in pj["manipulated_regions"][:3])
    arts = ", ".join(sorted({a for d in pj["manipulated_regions"][:3] for a in d["artifacts"]})[:3])
    return f"{FAKE_HDR} Manipulated regions: {regs}. Artifacts: {arts}."

MLLM_PROMPT = ("Given the isolated facial image and its extracted SRM high-frequency edge map, analyze structural and "
               "texture anomalies across landmark regions (mouth, eyes, nose, boundary). Output a standardized JSON object "
               "specifying manipulated regions, identified artifact types, and forensic reasoning.")

# ---------------- build items (notebook §7, real-data version) ----------------
def prepare_items(faces, domain, tag):
    items, t0 = [], time.time()
    for i in range(0, len(faces), 32):
        chunk = faces[i:i + 32]
        A = analyzer.analyze_batch([f["img"] for f in chunk], [f["lm"] for f in chunk])
        for j, (f, a) in enumerate(zip(chunk, A)):
            pj = analyzer.prompt(a, f"{tag}_{i+j}")
            items.append(dict(img=f["img"], lm=f["lm"], label=f["label"], prompt=pj, fused=a["fused"].astype(np.float16),
                              features=analyzer.feature_vector(a), fusion_text=fusion_text(pj), align_text=align_text(pj, f["label"]),
                              domain=domain, method=f["method"], video=f["video"],
                              gt=None, region="unknown"))
        if (i // 32) % 25 == 0: print(f"  {tag}: {len(items)}/{len(faces)}  ({time.time()-t0:.0f}s)", flush=True)
    return items


# ===================== loaders: restore everything from Drive in seconds =====================
def load_faces(name):
    out = []
    for f in sorted(CACHE.glob(f"{name}_*.npz")):
        z = np.load(f); imgs, lms, lab, met, vid = z["imgs"], z["lms"], z["label"], z["method"], z["video"]
        out += [dict(img=imgs[i], lm=lms[i], label=int(lab[i]), method=str(met[i]), video=str(vid[i])) for i in range(len(lab))]
    return out

def load_all(version="v2"):
    """Returns FACES (aligned crops) and DATA (items with prompts/features), rebuilt from Drive without recomputation.
    version="v2": two-sided priors (default). version="v1": original one-sided priors."""
    FACES = {s: load_faces(s) for s in ("train", "val", "test", "unseen")}
    mf = CK / ("items_meta_v2.pkl" if version == "v2" else "items_meta.pkl")
    print("loading prompts from", mf.name)
    meta = pickle.load(open(mf, "rb"))
    DATA = {}
    for s in FACES:
        assert len(FACES[s]) == len(meta[s]), f"{s}: cache and meta differ in length"
        assert all(f["label"] == m["label"] for f, m in zip(FACES[s], meta[s])), f"{s}: label mismatch"
        DATA[s] = [dict(img=f["img"], lm=f["lm"], fused=None, gt=None, **m) for f, m in zip(FACES[s], meta[s])]
    for s, v in DATA.items():
        print(f"{s:7s} {len(v):5d} items | real {sum(x['label']==0 for x in v)} fake {sum(x['label']==1 for x in v)}")
    return FACES, DATA

def load_priors():
    """Only needed to analyse NEW images or recompute pseudo-masks. Returns (priors, analyzer)."""
    ck = torch.load(CK / "priors.pt", map_location=DEVICE, weights_only=False)
    p = ForensicPriors(); p.ae = FaceAE().to(DEVICE); p.ae.load_state_dict(ck["ae"]); p.ae.eval()
    p.mu, p.sd = ck["mu"], ck["sd"]
    a = ForensicAnalyzer(p); a.reg_thr, a.prior_thr = ck["reg_thr"], ck["prior_thr"]
    if (CK / "thr_v2.pkl").exists():                      # thresholds calibrated for the two-sided (v2) priors
        t = pickle.load(open(CK / "thr_v2.pkl", "rb")); a.reg_thr, a.prior_thr = t["reg_thr"], t["prior_thr"]
    return p, a

print("refree_lib loaded | device:", DEVICE)

# ===================== Step 4: three-branch CLIP (build / train / evaluate) =====================
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, accuracy_score, f1_score

cfg.CLIP_NAME = "openai/clip-vit-base-patch16"
cfg.EPOCHS, cfg.BATCH, cfg.LR_CLIP, cfg.LR_HEAD = 8, 16, 1e-5, 1e-4
cfg.FP16, cfg.GRAD_CKPT = True, False
MEAN, STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)
TOK = None

class HFTokenizer:
    def __init__(self, name, max_len=77):
        from transformers import CLIPTokenizer
        self.tok, self.max_len = CLIPTokenizer.from_pretrained(name), max_len
    def __call__(self, texts):
        o = self.tok(list(texts), padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return o["input_ids"], o["attention_mask"]

class ThreeBranchCLIP(nn.Module):
    def __init__(self, clip, d_fuse=128, n_heads=4):
        super().__init__()
        self.clip = clip
        dv, dt, dp = clip.config.vision_config.hidden_size, clip.config.text_config.hidden_size, clip.config.projection_dim
        self.img_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(dp, 1))                     # branch 1: L_img
        self.vq, self.tkv = nn.Linear(dv, d_fuse), nn.Linear(dt, d_fuse)                      # branch 3: L_fusion
        self.xattn = nn.MultiheadAttention(d_fuse, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_fuse)
        self.fus_head = nn.Sequential(nn.Linear(d_fuse, d_fuse), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_fuse, 1))
        if cfg.GRAD_CKPT: clip.gradient_checkpointing_enable()
    def encode_image(self, px):
        v = self.clip.vision_model(pixel_values=px)
        return v.last_hidden_state, self.clip.visual_projection(v.pooler_output)
    def encode_text(self, ids, mask):
        t = self.clip.text_model(input_ids=ids, attention_mask=mask)
        return t.last_hidden_state, self.clip.text_projection(t.pooler_output)
    def forward(self, px, a_ids, a_mask, f_ids, f_mask):
        tokens, v_emb = self.encode_image(px)
        _, a_emb = self.encode_text(a_ids, a_mask)               # alignment caption (branch 2: L_align)
        f_tok, _ = self.encode_text(f_ids, f_mask)               # label-free heuristic text
        q = self.vq(tokens); kv = self.tkv(f_tok)
        att, _ = self.xattn(q, kv, kv, key_padding_mask=(f_mask == 0))
        fused = self.ln(q + att).mean(1)
        return dict(v_emb=v_emb, a_emb=a_emb, img_logit=self.img_head(v_emb).squeeze(-1),
                    fus_logit=self.fus_head(fused).squeeze(-1), logit_scale=self.clip.logit_scale)

def info_nce_soft(v, t, scale, texts):
    v, t = F.normalize(v, dim=-1), F.normalize(t, dim=-1)
    logits = scale.exp().clamp(max=100) * v @ t.T
    same = torch.tensor([[a == b for b in texts] for a in texts], device=v.device, dtype=torch.float32)
    P = same / same.sum(1, keepdim=True)
    l1 = -(P * F.log_softmax(logits, 1)).sum(1).mean()
    l2 = -((same / same.sum(0, keepdim=True)).T * F.log_softmax(logits.T, 1)).sum(1).mean()
    return 0.5 * (l1 + l2)

class ForgeryDS(Dataset):
    def __init__(self, items, train=False):
        self.items, self.train = items, train
        self.mean, self.std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; im = it["img"]
        if self.train and random.random() < 0.5: im = im[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, it["label"], it["align_text"], it["fusion_text"]

def collate(batch):
    x, y, at, ft = zip(*batch)
    a_ids, a_mask = TOK(at); f_ids, f_mask = TOK(ft)
    return torch.stack(x), torch.tensor(y, dtype=torch.float32), list(at), a_ids, a_mask, f_ids, f_mask

CLASS_TXT = [REAL_HDR, FAKE_HDR]
@torch.no_grad()
def predict(model, items, bs=32):
    model.eval(); dl = DataLoader(ForgeryDS(items), batch_size=bs, collate_fn=collate)
    c_ids, c_mask = TOK(CLASS_TXT); c_ids, c_mask = c_ids.to(DEVICE), c_mask.to(DEVICE)
    _, c_emb = model.encode_text(c_ids, c_mask); c_emb = F.normalize(c_emb.float(), dim=-1)
    P = {"img": [], "align": [], "fusion": []}; Y = []
    for x, y, at, a_ids, a_mask, f_ids, f_mask in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            out = model(x.to(DEVICE), a_ids.to(DEVICE), a_mask.to(DEVICE), f_ids.to(DEVICE), f_mask.to(DEVICE))
        sims = model.clip.logit_scale.exp().clamp(max=100) * F.normalize(out["v_emb"].float(), dim=-1) @ c_emb.T
        P["img"] += torch.sigmoid(out["img_logit"].float()).tolist()
        P["fusion"] += torch.sigmoid(out["fus_logit"].float()).tolist()
        P["align"] += sims.softmax(-1)[:, 1].tolist(); Y += y.tolist()
    P = {k: np.array(v) for k, v in P.items()}
    P["ensemble"] = (P["img"] + P["align"] + P["fusion"]) / 3
    return P, np.array(Y)

def eer_score(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2)
def metrics(y, s):
    return dict(AUC=roc_auc_score(y, s), EER=eer_score(y, s), ACC=accuracy_score(y, s > 0.5), F1=f1_score(y, s > 0.5))

def build_model():
    global TOK
    from transformers import CLIPModel
    TOK = HFTokenizer(cfg.CLIP_NAME)
    model = ThreeBranchCLIP(CLIPModel.from_pretrained(cfg.CLIP_NAME)).to(DEVICE)
    print(f"ThreeBranchCLIP: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters on {DEVICE}")
    return model

def train_model(model, train_items, val_items, epochs=None, bs=None):
    epochs, bs = epochs or cfg.EPOCHS, bs or cfg.BATCH
    t0 = time.time(); predict(model, val_items[::13])                       # 64-item smoke test of the full pipeline
    print(f"smoke test OK ({time.time()-t0:.1f}s). Training {epochs} epochs on {len(train_items)} faces ...", flush=True)
    clip_params = list(model.clip.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("clip.")]
    opt = torch.optim.AdamW([{"params": clip_params, "lr": cfg.LR_CLIP}, {"params": head_params, "lr": cfg.LR_HEAD}], weight_decay=1e-2)
    steps = epochs * (len(train_items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.LR_CLIP, cfg.LR_HEAD], total_steps=steps, pct_start=0.15)
    use_amp = cfg.FP16 and DEVICE == "cuda"
    try: scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception: scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dl = DataLoader(ForgeryDS(train_items, train=True), batch_size=bs, shuffle=True, collate_fn=collate, drop_last=True)
    hist, best = [], -1.0
    for ep in range(epochs):
        t_ep = time.time(); model.train(); agg = np.zeros(4); n = 0
        for x, y, at, a_ids, a_mask, f_ids, f_mask in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            a_ids, a_mask, f_ids, f_mask = [t.to(DEVICE) for t in (a_ids, a_mask, f_ids, f_mask)]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                o = model(x, a_ids, a_mask, f_ids, f_mask)
                l_img = F.binary_cross_entropy_with_logits(o["img_logit"].float(), y)
                l_aln = info_nce_soft(o["v_emb"].float(), o["a_emb"].float(), o["logit_scale"], at)
                l_fus = F.binary_cross_entropy_with_logits(o["fus_logit"].float(), y)
                loss = l_img + l_aln + l_fus                                   # Eq. (1): L_total = L_img + L_align + L_fusion
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            agg += np.array([loss.item(), l_img.item(), l_aln.item(), l_fus.item()]) * len(y); n += len(y)
        P, Y = predict(model, val_items)
        row = dict(epoch=ep + 1, loss=agg[0] / n, L_img=agg[1] / n, L_align=agg[2] / n, L_fusion=agg[3] / n,
                   val_AUC_img=roc_auc_score(Y, P["img"]), val_AUC_ens=roc_auc_score(Y, P["ensemble"]))
        hist.append(row)
        if row["val_AUC_ens"] > best:                                          # keep the best checkpoint on Drive
            best = row["val_AUC_ens"]; torch.save(model.state_dict(), CK / "clip3_best.pt")
        print("  ".join(f"{k}={v:.3f}" if k != "epoch" else f"epoch {v:2d}" for k, v in row.items()),
              f"| {time.time()-t_ep:.0f}s", flush=True)
    pd.DataFrame(hist).to_csv(CK / "train_history.csv", index=False)
    load_best(model)
    return pd.DataFrame(hist)

def load_best(model):
    model.load_state_dict(torch.load(CK / "clip3_best.pt", map_location=DEVICE)); model.eval()
    print("restored best checkpoint from Drive")
    return model

def evaluate_all(model, DATA):
    heads = ["img", "align", "fusion", "ensemble"]; tabs, preds = {}, {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a_, (split, nm) in zip(ax, [("test", "intra-domain (FF++ c23 test)"), ("unseen", "cross-dataset (Celeb-DF v2, zero-shot)")]):
        P, Y = predict(model, DATA[split]); preds[split] = (P, Y)
        tab = pd.DataFrame({h: metrics(Y, P[h]) for h in heads}).T; tabs[split] = tab
        print(f"\n{nm}:\n{tab.round(3).to_string()}")
        for h in heads:
            f, t, _ = roc_curve(Y, P[h]); a_.plot(f, t, label=f"{h} (AUC {roc_auc_score(Y, P[h]):.2f})")
        a_.plot([0, 1], [0, 1], "k:"); a_.set_title(f"ROC - {nm}", fontsize=9); a_.set_xlabel("FPR"); a_.set_ylabel("TPR"); a_.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    P, Y = preds["test"]; meth = np.array([x["method"] for x in DATA["test"]])
    print("\nFF++ test, per manipulation method (that method's fakes vs all real faces), AUC:")
    for m in sorted(set(meth) - {"real"}):
        sel = (meth == m) | (Y == 0)
        print(f"  {m:15s} img {roc_auc_score(Y[sel], P['img'][sel]):.3f} | ensemble {roc_auc_score(Y[sel], P['ensemble'][sel]):.3f}")
    pd.concat(tabs, names=["split", "head"]).to_csv(CK / "results_clip.csv")
    np.savez(CK / "preds.npz", **{f"{s}_{h}": preds[s][0][h] for s in preds for h in heads}, **{f"{s}_y": preds[s][1] for s in preds})
    print("\nsaved results_clip.csv and preds.npz to Drive")
    return tabs
