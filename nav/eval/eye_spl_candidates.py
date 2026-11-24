"""Candidate instances for eye_spl's --instances TSV: for every eye-S episode that Habitat scores a failure, the semantic
instances whose bounding box is within 1 m (horizontal) of the final position on the agent's floor, nearest first.
An episode step-identical to a reference run (ledger verifier inherit-*) reuses the reference's pick.

  python -m nav.eval.eye_spl_candidates RUN [RUN ...] --out TSV          (apexnav container: /opt/venv/apexnav/bin/python)"""
import argparse, csv, glob, json, os
import numpy as np

from nav.eval.closer_attrib import jl
from nav.eval.eye_spl import LABELS, RUNS, meta_of, navmesh, objects, path, start_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", choices=list(RUNS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    led = [r for r in csv.DictReader(open(LABELS + "ledger.tsv"), delimiter="\t") if r["run"] in a.runs and r["label"] == "S"]
    rows, scene, sim, objs = [], None, None, None
    todo = []
    for r in led:
        d = f"{RUNS[r['run']][0]}/epi{r['episode']}"
        E = jl(d + "/episodes.jsonl")[-1]
        if not E["success"]:
            todo.append((E["scene"], r, glob.glob(d + "/records/*/*/steps.jsonl")[0], E))
    for sc, r, sp, E in sorted(todo, key=lambda t: t[0]):
        if sc != scene:
            if sim:
                sim.close()
            scene, sim, objs = sc, navmesh(sc), objects(sc)
        steps, meta = jl(sp), meta_of(sp, E)
        start, rot = start_pose(meta, E)
        _, final = path(steps, start, rot, sim.pathfinder)
        cand = []
        for i, (name, v) in objs.items():
            lo, hi = v.min(0), v.max(0)
            if hi[1] < final[1] - 0.3 or lo[1] > final[1] + 2.0:  # other floor
                continue
            dh = np.hypot(max(lo[0] - final[0], 0, final[0] - hi[0]), max(lo[2] - final[2], 0, final[2] - hi[2]))
            if dh <= 1.0:
                cand.append((dh, i, name))
        cand.sort()
        rows.append(dict(run=r["run"], episode=r["episode"], target=E["target_label"], verifier=r["verifier"], note=r["note"],
                         candidates="; ".join(f"{i}:{n}:{dh:.2f}" for dh, i, n in cand)))
    with open(a.out, "w") as f:
        w = csv.DictWriter(f, ["run", "episode", "target", "verifier", "note", "candidates"], delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(len(rows), "episodes ->", a.out)


if __name__ == "__main__":
    main()
