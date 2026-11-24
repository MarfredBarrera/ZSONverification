"""Evidence quality at the commit of each eye-labelled false stop (ledger FS-perc). CPU only; definitions in
outputs/fp_severity/evidence/README.md.   /home/marfred/.conda/envs/habitat/bin/python -m nav.eval.fp_evidence
Writes outputs/fp_severity/evidence/{items.tsv,medians.tsv,tests.tsv}."""
import csv, glob, json, math, os
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu
from nav.eval.fp_severity import RUNS, A, OUT, record, rows, LEDGER

RUNS["hm3d_split1000"] = (A + "hm3d_split1000/gate", "apexnav", "hm3d", "gate")
USE = ["hm3d_raw1000", "hm3d_split1000", "hm3d_rec_raw1000", "hm3d_rec_zson1000"]
OUTD = OUT + "evidence/"
IMG_PX, HFOV, MERGE_M = 640 * 480, math.radians(79), 1.0
COLS = ["run", "episode", "planner", "arm", "target", "commit_step", "commit_src", "inst_id", "px_frac", "dist_m", "n_views",
        "visible_frac", "conf", "bearing_deg", "px_check", "missing"]


def dom(det):
    si = det.get("semantic_instances") or []
    return max(si, key=lambda i: i["pixels"]) if si else None


def all_dets(s):
    return s["detections"] + s.get("dropped_detections", [])   # VLFM also logs sub-threshold target-class detections


def commit_detection(run, ep, rec, d, S):
    """(T, step of the committed view, detection, src, px_check): T = the decision step (steps.jsonl numbering) that bounds n_views."""
    by = {s["step"]: s for s in S}
    if RUNS[run][3] == "raw":
        fs = [s["final_state"] for s in S]; t = len(fs)
        while t > 0 and fs[t - 1] != 0: t -= 1
        T = S[t]["step"]   # first step of the last stretch with final_state != EXPLORE
        pcs = [s for s in S[t:] if s["planner_commit"]]
        if pcs: pc = pcs[0]["planner_commit"]
        else:   # commit not logged: the planner picks the most confident target cluster ('highest_confident_primary_cluster')
            cl = [c for c in S[t]["object_fusion"]["clusters"] if c["best_label"] == 0 and c.get("is_confident", True)]
            if not cl: raise LookupError("no commit logged and no confident target cluster at the stop (committed object not identifiable)")
            pc = dict(centroid=max(cl, key=lambda c: c["confidence"])["centroid"]); nolog = "_clusterrule"
        det_at = lambda st: by[st]["detections"]
        if pcs and pc.get("matched_detection_index") is not None and T == pcs[0]["step"]:
            return T, T, by[T]["detections"][pc["matched_detection_index"]], "commit_step", None
        # commit on a remembered cluster: latest earlier view whose detection lies within MERGE_M of the committed centroid
        for st in range(T, 0, -1):
            c = [x for x in by[st]["detections"] if x.get("world_centroid") and math.dist(x["world_centroid"], pc["centroid"]) <= MERGE_M]
            if c: return T, st, min(c, key=lambda x: math.dist(x["world_centroid"], pc["centroid"])), f"last_seen_{T - st}_steps_before" + ("_clusterrule" if not pcs else ""), None
        raise LookupError("committed cluster never seen in a detection")
    a = [r for r in map(json.loads, open(d + "/verification/decisions.jsonl")) if r.get("state") == 1][-1]
    off = 0 if RUNS[run][1] == "apexnav" else 1   # gate step N = steps.jsonl step N (ApexNav) / N+1 (VLFM), checked on seed mask_px
    T = a["step"] + off
    if a.get("approach") is None:
        n, k, px = a["evidence"]["step"], a["evidence"]["index"], a["ground_truth"]["semantic_pixels"]
        det = by[n + off]["detections"][k]
    else:
        v = a["approach"]["views"][-1]
        if v >= 0: raise LookupError("last counted view is a live approach frame (no semantic ids logged)")
        ap = max(glob.glob(f"{d}/verification/approach/epi*_c{a['cluster']['id']}_s*"), key=lambda p: int(p.rsplit("_s", 1)[1]))
        seed = [e for e in map(json.loads, open(ap + "/approach.jsonl")) if e.get("event") == "seed" and e["step"] == v][-1]
        n, px = seed["bank_step"], seed["mask_px"]
        c = [x for x in by[n + off]["detections"] if x.get("world_centroid")]
        det = min(c, key=lambda x: math.dist(x["world_centroid"], a["approach"]["object"]))
    chk = sum(i["pixels"] for i in det.get("semantic_instances") or []) - px
    return T, n + off, det, "evidence_view", chk


def item(r):
    run, ep = r["run"], r["episode"]
    try: rec, d = record(run, ep)
    except FileNotFoundError: return dict(run=run, episode=ep, planner=RUNS[run][1], arm=RUNS[run][3], missing="no records (frames) stored")
    S = [json.loads(l) for l in open(rec + "steps.jsonl")]
    o = dict(run=run, episode=ep, planner=RUNS[run][1], arm=RUNS[run][3], target=json.load(open(rec + "meta.json"))["target_label"])
    try: T, st, det, src, chk = commit_detection(run, ep, rec, d, S)
    except LookupError as e: o["missing"] = str(e); return o
    inst = dom(det)
    if inst is None: o["missing"] = "committed detection has no semantic instance under its mask"; return o
    by = {s["step"]: s for s in S}
    seen = {s["step"]: max((i["pixels"] for x in all_dets(s) for i in x.get("semantic_instances") or [] if i["id"] == inst["id"]), default=0) for s in S}
    o.update(commit_step=st, commit_src=src, inst_id=inst["id"], px_frac=round(inst["pixels"] / IMG_PX, 5), conf=round(det["score"], 4),
             n_views=sum(1 for s, p in seen.items() if p > 0 and s <= T), visible_frac=round(inst["pixels"] / max(seen.values()), 3), px_check=chk)
    if det.get("world_centroid"):
        x, y, yaw = by[st]["pose"]; cx, cy = det["world_centroid"]
        o["dist_m"] = round(math.hypot(cx - x, cy - y), 3)
        o["bearing_deg"] = round(math.degrees((math.atan2(cy - y, cx - x) - yaw + math.pi) % (2 * math.pi) - math.pi), 1)
    else: o["missing"] = "no world_centroid (mask has no valid depth): dist_m empty"
    return o


def tests(items):
    g = lambda run, k: np.array([float(i[k]) for i in items if i["run"] == run and i.get(k) not in (None, "")])
    out = []
    for name, ra, rb, ka, kb in [("T1 vlfm raw>gate", "hm3d_rec_raw1000", "hm3d_rec_zson1000", "hm3d_rec_raw1000", "hm3d_rec_zson1000"),
                                 ("T2 apexnav gate>vlfm gate", "hm3d_split1000", "hm3d_rec_zson1000", None, None)]:
        for k, sign in [("px_frac", 1), ("n_views", 1), ("visible_frac", 1), ("dist_m", -1)]:   # sign -1: hypothesis is first < second
            a, b = sign * g(ra, k), sign * g(rb, k)
            u = mannwhitneyu(a, b, alternative="greater")
            out.append(dict(test=name, metric=k, direction="first>second" if sign > 0 else "first<second", n1=len(a), n2=len(b),
                            median1=round(float(np.median(sign * a)), 4), median2=round(float(np.median(sign * b)), 4),
                            auc=round(u.statistic / (len(a) * len(b)), 3), p_one_sided=round(float(u.pvalue), 4)))
    return out


def write(path, rs, cols):
    with open(path, "w") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n", restval=""); w.writeheader(); w.writerows(rs)


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    items = [item(r) for r in rows(LEDGER) if r["label"] == "FS-perc" and r["run"] in USE]
    write(OUTD + "items.tsv", items, COLS)
    med = []
    for run in USE:
        it = [i for i in items if i["run"] == run]
        m = dict(run=run, planner=RUNS[run][1], arm=RUNS[run][3], n=len(it))
        for k in ["px_frac", "dist_m", "n_views", "visible_frac", "conf"]:
            v = [float(i[k]) for i in it if i.get(k) not in (None, "")]
            m[k] = f"{np.median(v):.4g} (n={len(v)})"
        med.append(m)
    write(OUTD + "medians.tsv", med, list(med[0]))
    t = tests(items); write(OUTD + "tests.tsv", t, list(t[0]))
    for r in med + t: print(r)
