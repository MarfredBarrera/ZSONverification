"""Human-similarity severity of eye-labelled false stops from the THINGS SPoSE similarity matrix.

  python -I -m nav.eval.fp_things      (habitat conda env: numpy+scipy)  ->  outputs/fp_severity/things/items.tsv + analysis.txt

S = shipped THINGS `spose_similarity.mat` (1854x1854, concept order of concepts-metadata_things.tsv); verified equal to the
canonical SPoSE formula on the 66-d embedding (mean over k of the softmax probability that (i,j) is the chosen pair).
severity = 1 - S, no rescaling. Items: ledger FS-perc rows of RUNS. Goal = failure_selected `goal`; object text = eye label
`true_name` (fp_severity/labels.tsv) when present, else failure_selected `selected`. Concept map: things/mapping.tsv."""
import csv, io, itertools, sys
import numpy as np
import scipy.io
from scipy.stats import mannwhitneyu

ROOT = __file__.rsplit("/nav/", 1)[0]
LAB, OUT = ROOT + "/outputs/apexnav_online/labels/", ROOT + "/outputs/fp_severity/"
T = OUT + "things/"
# run -> (planner, dataset, arm)
RUNS = {"hm3d_raw1000": ("apexnav", "hm3d", "raw"), "hm3d_split1000": ("apexnav", "hm3d", "gate"),
        "hm3d_rec_raw1000": ("vlfm", "hm3d", "raw"), "hm3d_rec_zson1000": ("vlfm", "hm3d", "gate"),
        "ovon_raw200": ("apexnav", "ovon", "raw"), "ovon_split200": ("apexnav", "ovon", "gate")}
REF = [("couch", "armchair"), ("couch", "sofa"), ("bed", "dog bed"), ("couch", "loveseat"), ("couch", "chair"), ("couch", "sofa bed"), ("chair", "bench"),
       ("bed", "sofa bed"), ("bed", "couch"), ("toilet", "trashcan"), ("television", "mirror"), ("television", "wall"),
       ("television", "photograph"), ("plant", "painting"), ("chair", "stool"), ("chair", "recliner"), ("bed", "chair"),
       ("chair", "bed"), ("toilet", "bench"), ("bed", "car"), ("couch", "wall")]


def rows(p): return list(csv.DictReader(open(p), delimiter="\t"))


def main():
    names = [r["Word"] for r in rows(T + "download/concepts-metadata_things.tsv")]
    ix = {n: i for i, n in enumerate(names) if names.count(n) == 1}  # duplicated words (tank, bow, ...) unused
    S = scipy.io.loadmat(T + "download/spose_similarity.mat")["spose_sim"]
    assert S.shape == (len(names),) * 2
    mp = {(r["kind"], r["text"]): r["concept"] for r in rows(T + "mapping.tsv")}
    merge = {(r["old_run"], r["episode"]): r["new_run"] for r in rows(LAB + "run_merge_2026-11-19.tsv")}
    sel = {(merge.get((r["run"], r["ep"]), r["run"]), r["ep"]): r for r in rows(LAB + "failure_selected.tsv")}
    eye = {(r["run"], r["episode"]): r["true_name"] for r in rows(OUT + "labels.tsv")}
    items = []
    for r in rows(LAB + "ledger.tsv"):
        if r["label"] != "FS-perc" or r["run"] not in RUNS: continue
        k = (r["run"], r["episode"])
        goal, obj = sel[k]["goal"], eye.get(k, sel[k]["selected"])
        gc, oc = mp[("goal", goal)], mp[("obj", obj)]
        pl, ds, arm = RUNS[r["run"]]
        if "UNMAPPED" in (gc, oc): s, status = np.nan, "UNMAPPED"
        else: s, status = S[ix[gc], ix[oc]], "SAME_CONCEPT" if gc == oc else "ok"
        items.append(dict(planner=pl, dataset=ds, run=r["run"], arm=arm, episode=r["episode"], goal=goal, object_text=obj,
                          goal_concept=gc, object_concept=oc, S=s, severity=1 - s, status=status,
                          obj_source="eye" if k in eye else "selected"))
    cols = list(items[0])
    w = csv.writer(open(T + "items.tsv", "w"), delimiter="\t"); w.writerow(cols)
    for it in items: w.writerow([f"{it[c]:.4f}" if isinstance(it[c], float) else it[c] for c in cols])

    log = io.StringIO()
    def p(*a): print(*a); print(*a, file=log)
    p("items", len(items), " status:", {s: sum(i["status"] == s for i in items) for s in ("ok", "SAME_CONCEPT", "UNMAPPED")})
    def sev(pl, arm, ds, same):
        return np.array([i["severity"] for i in items if (i["planner"], i["arm"], i["dataset"]) == (pl, arm, ds)
                         and (i["status"] == "ok" or (same and i["status"] == "SAME_CONCEPT"))])
    def test(a, b, alt, label, same):
        x, y = sev(*a, same), sev(*b, same)
        u, pv = mannwhitneyu(x, y, alternative=alt)
        p(f"  {label}: n={len(x)} vs {len(y)}  mean sev {x.mean():.3f} vs {y.mean():.3f}  median {np.median(x):.3f} vs "
          f"{np.median(y):.3f}  MWU p({alt})={pv:.3g}  AUC P(first>second)={u / (len(x) * len(y)):.3f}")
    for same in (True, False):
        p(f"\n== severity tests; SAME_CONCEPT items {'kept at S=matrix diagonal=1 (severity 0)' if same else 'excluded'}; UNMAPPED excluded")
        for pl in ("apexnav", "vlfm"):
            test((pl, "raw", "hm3d"), (pl, "gate", "hm3d"), "greater", f"hm3d {pl} raw > gate", same)
        test(("apexnav", "raw", "ovon"), ("apexnav", "gate", "ovon"), "greater", "ovon apexnav raw > gate", same)
        test(("vlfm", "gate", "hm3d"), ("apexnav", "gate", "hm3d"), "two-sided", "hm3d gate vlfm vs apexnav", same)
    p("\n== counts per run (n, ok, same, unmapped) and mean S over ok+same")
    for run in RUNS:
        g = [i for i in items if i["run"] == run]; v = [i["S"] for i in g if i["status"] != "UNMAPPED"]
        p(f"  {run:20s} n={len(g)} ok={sum(i['status'] == 'ok' for i in g)} same={sum(i['status'] == 'SAME_CONCEPT' for i in g)} "
          f"unmapped={sum(i['status'] == 'UNMAPPED' for i in g)} meanS={np.mean(v):.3f}")
    p("\n== mean S by (planner, dataset, arm) and goal->object concept (n>=2; UNMAPPED excluded)")
    for key in sorted({(i["planner"], i["dataset"], i["arm"]) for i in items}):
        g = {}
        for i in items:
            if (i["planner"], i["dataset"], i["arm"]) == key and i["status"] != "UNMAPPED":
                g.setdefault((i["goal_concept"], i["object_concept"]), []).append(i["S"])
        top = sorted(g.items(), key=lambda kv: -len(kv[1]))[:8]
        p(" ", key, "; ".join(f"{a}->{b} n={len(v)} S={np.mean(v):.3f}" for (a, b), v in top))
    p("\n== reference pairs (THINGS S, severity=1-S); chance-ish level for unrelated pairs is about 1/3")
    for a, b in REF:
        p(f"  {a:11s} {b:11s} " + (f"S={S[ix[a], ix[b]]:.3f}  severity={1 - S[ix[a], ix[b]]:.3f}" if a in ix and b in ix else "not a (unique) THINGS concept"))
    off = S[np.triu_indices(len(S), 1)]
    p(f"  all pairs: mean {off.mean():.3f} p5 {np.percentile(off, 5):.3f} p50 {np.median(off):.3f} p95 {np.percentile(off, 95):.3f}")
    open(T + "analysis.txt", "w").write(log.getvalue())


if __name__ == "__main__":
    main()
