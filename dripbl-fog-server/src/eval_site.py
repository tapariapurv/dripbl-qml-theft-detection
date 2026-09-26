"""Clip accuracy of a model on a custom site's fixed hold-out: eval_site.py <model> <site>."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import cross_data as C

model, site = sys.argv[1], sys.argv[2]
fn, thr = C.artifact_margin_fn(model)
d = C.custom_part(site, "holdout")
lab, sc = C.clip_scores(fn(d["rgb"], d.get("motion"), d["vid"]), d["vid"], d["y"])
print(site, model, C.report(lab, sc, thr))
