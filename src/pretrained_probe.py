"""Probe: frozen ImageNet backbone instead of a from-scratch CNN.

Every scratch-trained backbone so far plateaued at clip AUC ~0.87-0.91 because
it overfits the training *scenes* -- Train and Test share no source videos, so
memorised scenes buy nothing. A frozen ImageNet backbone cannot overfit this
dataset at all (it never saw it), and it needs no training: one forward pass and
the rest of the pipeline is unchanged.

Pipeline kept intact: frames -> CNN features -> PCA(128) latent -> PLS/PCA(4)
angles in [-pi, pi] -> ZZFeatureMap fidelity kernel -> SVC(precomputed).

    .venv/bin/python3 -u src/pretrained_probe.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

FRAMES_PER_VIDEO = 30
TEST_FRAMES_PER_VIDEO = 60
LATENT_DIM = 128
SVM_FIT_FRAMES = 3000   # the frame-level Gram matrix is O(n^2): 29k frames would be ~7 GB


def build_backbone(name: str = "mobilenet_v3_small") -> torch.nn.Module:
    from torchvision.models import (MobileNet_V3_Small_Weights, ResNet18_Weights,
                                    mobilenet_v3_small, resnet18)
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        return model.eval()
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()).eval()


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.inference_mode()
def extract(model: torch.nn.Module, images: np.ndarray, batch: int = 256) -> np.ndarray:
    out = []
    for start in range(0, len(images), batch):
        x = torch.from_numpy(images[start:start + batch]).permute(0, 3, 1, 2).float().div_(255.0)
        out.append(model((x - _MEAN) / _STD).numpy())
    return np.concatenate(out)


def qsvm_scores(lat_fit: np.ndarray, y_fit: np.ndarray, lat_val: np.ndarray, y_val: np.ndarray,
                lat_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Fit PLS/PCA(4) + quantum-kernel SVC; select on validation, score both splits."""
    best = (-1.0, None, None, None)
    projectors = {m: PCAAngleProjector(n_components=FEATURE_DIM, mode=m).fit(lat_fit, y_fit)
                  for m in ("pca", "pls")}
    for mode, proj in projectors.items():
        for scale in T.ANGLE_SCALES:
            proj.angle_scale = scale
            e_fit, e_val = encode(proj.transform(lat_fit)), encode(proj.transform(lat_val))
            k_fit, k_val = kernel_matrix(e_fit), kernel_matrix(e_val, e_fit)
            for c in T.C_GRID:
                svc = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(k_fit, y_fit)
                acc = balanced_accuracy_score(y_val, svc.predict(k_val))
                if acc > best[0]:
                    best = (float(acc), mode, scale, c)
    val_acc, mode, scale, c = best
    proj = projectors[mode]
    proj.angle_scale = scale
    e_fit = encode(proj.transform(lat_fit))
    clf = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(kernel_matrix(e_fit), y_fit)
    val_m = clf.decision_function(kernel_matrix(encode(proj.transform(lat_val)), e_fit))
    test_m = clf.decision_function(kernel_matrix(encode(proj.transform(lat_test)), e_fit))
    return val_m, test_m, (val_acc, mode, scale, c)


def main() -> None:
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    t0 = time.time()
    cfg = E.Config("pretrained", frames_per_video=FRAMES_PER_VIDEO,
                   test_frames_per_video=TEST_FRAMES_PER_VIDEO)
    data = E.get_data(cfg)
    x_tr, y_tr, vid_tr = data["train"]
    x_va, y_va, vid_va = data["val"]
    x_te, y_te, vid_te, _ = data["test"]
    print(f"[1/4] data: train={len(y_tr)} val={len(y_va)} test={len(y_te)} ({time.time() - t0:.0f}s)", flush=True)

    results = {}
    for name in ("mobilenet_v3_small", "resnet18"):
        t1 = time.time()
        model = build_backbone(name)
        f_tr, f_va, f_te = (extract(model, x) for x in (x_tr, x_va, x_te))
        print(f"[2/4] {name}: features {f_tr.shape[1]}-d in {time.time() - t1:.0f}s", flush=True)

        # keep the documented 128-d latent stage
        pca = PCA(n_components=LATENT_DIM, random_state=T.SEED).fit(f_tr)
        lat_tr, lat_va, lat_te = pca.transform(f_tr), pca.transform(f_va), pca.transform(f_te)

        # (a) frame-level QSVM, clip verdict by aggregating frame margins
        sub = np.random.default_rng(T.SEED).permutation(len(y_tr))[:SVM_FIT_FRAMES]
        val_m, test_m, params = qsvm_scores(lat_tr[sub], y_tr[sub], lat_va, y_va, lat_te)
        best_frame = (-1.0, None, None)
        for how in E.AGGREGATIONS:
            vl, vs = E.aggregate(vid_va, y_va, val_m, how)
            thr, vacc = E.pick_threshold(vl, vs)
            best_frame = max(best_frame, (vacc, how, thr))
        _, how, thr = best_frame
        tl, ts = E.aggregate(vid_te, y_te, test_m, how)
        frame_path = E.clip_report(tl, ts, thr)
        frame_path["oracle_threshold_accuracy"] = E.pick_threshold(tl, ts)[1]
        frame_path["aggregation"] = how
        frame_path["selection"] = {"val_acc": params[0], "projector": params[1],
                                   "angle_scale": params[2], "C": params[3]}

        # (b) clip-descriptor QSVM
        d_tr, yc_tr, _ = E.clip_descriptors(lat_tr, vid_tr, y_tr)
        d_va, yc_va, _ = E.clip_descriptors(lat_va, vid_va, y_va)
        d_te, yc_te, _ = E.clip_descriptors(lat_te, vid_te, y_te)
        val_c, test_c, params_c = qsvm_scores(d_tr, yc_tr, d_va, yc_va, d_te)
        thr_c, _ = E.pick_threshold(yc_va, val_c)
        clip_path = E.clip_report(yc_te, test_c, thr_c)
        clip_path["oracle_threshold_accuracy"] = E.pick_threshold(yc_te, test_c)[1]
        clip_path["selection"] = {"val_acc": params_c[0], "projector": params_c[1],
                                  "angle_scale": params_c[2], "C": params_c[3]}

        # (c) stack the two score paths (they make different mistakes)
        def z(a: np.ndarray) -> np.ndarray:
            return (a - a.mean()) / (a.std() + 1e-9)
        tl2, ts2 = E.aggregate(vid_te, y_te, test_m, how)
        vl2, vs2 = E.aggregate(vid_va, y_va, val_m, how)
        stack_val = z(vs2) + z(val_c)
        stack_test = z(ts2) + z(test_c)
        thr_s, _ = E.pick_threshold(vl2, stack_val)
        stack_path = E.clip_report(tl2, stack_test, thr_s)
        stack_path["oracle_threshold_accuracy"] = E.pick_threshold(tl2, stack_test)[1]

        results[name] = {"frame_path": frame_path, "clip_path": clip_path, "stacked": stack_path}
        for label, r in results[name].items():
            print(f"  {name:20s} {label:11s} acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} "
                  f"auc={r['roc_auc']:.4f} rec={r['recall']:.3f} oracle={r['oracle_threshold_accuracy']:.4f}",
                  flush=True)

    out = Path(__file__).resolve().parent.parent / "reports" / "pretrained_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
