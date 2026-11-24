"""How the choice of frame changes false-stop severity (1 - percentile of SigLIP "a photo of a {goal}" logit among real-goal
reference stops scored with the SAME frame rule). Only the last ~30 rgb frames of an episode are stored.
  CUDA_VISIBLE_DEVICES=0 sam3d-env sam3d python -m nav.eval.fp_frames   logits (cached) -> scores.tsv, tests.txt, sev_by_frame.png
Rules: decision (decision_frame) | final (last stored) | win_mean / win_max (stored frames from the decision frame's rgb index on;
approach accept: approach frame + stored frames with index >= the approach folder's step; decision frame not stored: all stored frames)
| last5_mean | all30_mean."""
import csv, functools, glob, os, re
import numpy as np
from nav.eval import fp_dreamsim as dd, fp_severity as fs
from nav.eval.fp_dreamsim import pct

OUT = fs.ROOT + "/outputs/fp_severity/frames/"
RULES = ["decision", "final", "win_mean", "win_max", "last5_mean", "all30_mean"]
MEAN, MAX = np.mean, np.max


def read(p): return list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []


BAD = []   # truncated jpgs, dropped from every window and recorded in tests.txt


@functools.lru_cache(None)
def readable(p):
    from PIL import Image
    try: Image.open(p).load(); return True
    except OSError as e: BAD.append(f"{p.split('videos/')[-1].split('outputs/')[-1]} ({e})"); return False


def frames(run, ep):
    """(stored rgb [(index, path)], decision (path, n) or None, why-not)"""
    rec = fs.record(run, ep)[0]
    stored = [(int(os.path.basename(p)[:-4]), p) for p in sorted(glob.glob(rec + "rgb/*.jpg"))[-30:] if readable(p)]   # last 30 rgb indices (full records are cut to match)
    try: return stored, fs.decision_frame(run, ep), ""
    except FileNotFoundError as e: return stored, None, "decision frame not stored"


def window(run, ep):
    """frames of the window as [(index or None, path)], note"""
    stored, dec, why = frames(run, ep)
    if dec is None: return stored, why + ": window = all stored"
    p, n = dec
    if n is not None: return [x for x in stored if x[0] >= n], ""
    step = int(p.split("/frames/")[0].rsplit("_s", 1)[1])
    return [(None, p)] + [x for x in stored if x[0] >= step], "approach accept"


def score_logits(todo):
    """todo: [(run, ep, goal, [(idx, path)])] -> rows; SigLIP exactly as fp_dreamsim.score text_scores, batched, always released."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(4)
    m = AutoModel.from_pretrained(fs.MODEL, torch_dtype=torch.float16).cuda().eval(); pr = AutoProcessor.from_pretrained(fs.MODEL)
    out, emb = [], {}
    try:
        with torch.no_grad():
            for run, ep, goal, fr in todo:
                for i in range(0, len(fr), 1):
                    b = fr[i:i + 1]
                    x = pr(text=[f"a photo of a {goal}"], images=[Image.open(p).convert("RGB") for _, p in b], padding="max_length", return_tensors="pt").to("cuda")
                    x["pixel_values"] = x["pixel_values"].half()
                    o = m(**x); e = o.image_embeds.float().cpu().numpy().astype(np.float16)   # L2-normalized
                    for (idx, p), v, ei in zip(b, o.logits_per_image[:, 0].float().tolist(), e):
                        emb[p] = ei; out.append(dict(run=run, episode=ep, frame_path=p, rgb_index="" if idx is None else idx, goal=goal, logit=v))
        ls, lb = m.logit_scale.item(), m.logit_bias.item()
    finally:
        del m; torch.cuda.empty_cache()
    return out, emb, ls, lb


def rule_values(run, ep, L):
    """{rule: value or None}, notes. L = {frame_path: logit} of this episode."""
    stored, dec, why = frames(run, ep); w, wn = window(run, ep)
    S = [L[p] for _, p in stored]; W = [L[p] for _, p in w]; notes = [x for x in (why, wn) if x]
    return {"decision": L[dec[0]] if dec else None, "final": S[-1], "win_mean": MEAN(W), "win_max": MAX(W), "last5_mean": MEAN(S[-5:]), "all30_mean": MEAN(S)}, notes


def main():
    items, refs = read(dd.OUT + "scores.tsv"), read(dd.OUT + "ref_items.tsv"); dropped = []
    def has(r):
        try: fs.record(r["run"], int(r["episode"])); return True
        except FileNotFoundError: dropped.append(f"{r['run']} ep{r['episode']} (no stored frames)"); return False
    items, refs = [r for r in items if has(r)], [r for r in refs if has(r)]
    os.makedirs(OUT, exist_ok=True)
    cache = read(OUT + "logits.tsv"); have = {(r["run"], r["episode"], r["frame_path"]) for r in cache}
    need = {}
    for r in items + refs:
        k = (r["run"], r["episode"])
        if k in need: continue
        stored, dec, _ = frames(r["run"], int(r["episode"])); w, _ = window(r["run"], int(r["episode"]))
        allf = {p: i for i, p in stored}
        for i, p in w + ([(None, dec[0])] if dec else []): allf.setdefault(p, i)
        need[k] = (r["goal"], [(i, p) for p, i in allf.items() if (r["run"], r["episode"], p) not in have])
    todo = [(k[0], k[1], g, f) for k, (g, f) in need.items() if f]
    if todo:
        new, emb, ls, lb = score_logits(todo); cache += new
        P = sorted(emb); np.savez(OUT + "img_emb.npz", paths=np.array(P), emb=np.stack([emb[p] for p in P]), logit_scale=ls, logit_bias=lb)
        with open(OUT + "logits.tsv", "w") as f:
            w = csv.DictWriter(f, list(cache[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(cache)
    LG = {}
    for r in cache: LG.setdefault((r["run"], r["episode"]), {})[r["frame_path"]] = float(r["logit"])
    RV = {}
    for r in refs: RV[r["run"], r["episode"]] = rule_values(r["run"], int(r["episode"]), LG[r["run"], r["episode"]])[0]
    skip = {x: [] for x in RULES}; wl = []; res = []
    for it in items:
        v, notes = rule_values(it["run"], int(it["episode"]), LG[it["run"], it["episode"]]); wl.append(len(window(it["run"], int(it["episode"]))[0]))
        row = dict(planner=it["planner"], arm=it["arm"], run=it["run"], episode=it["episode"], goal=it["goal"])
        for rule in RULES:
            ref = [RV[r["run"], r["episode"]][rule] for r in refs if r["goal"] == it["goal"] and RV[r["run"], r["episode"]][rule] is not None]
            if v[rule] is None: row["sev_" + rule] = ""; skip[rule].append((it["run"], it["episode"], notes)); continue
            row["sev_" + rule] = 1 - pct(v[rule], ref)
        row["notes"] = "; ".join(notes); res.append(row)
    with open(OUT + "scores.tsv", "w") as f:
        w = csv.DictWriter(f, list(res[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(res)
    stats(res, skip, wl, dropped)


def stats(res, skip, wl, dropped):
    import scipy.stats as st, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rep = []; P = {"apexnav": ("hm3d_raw1000", "hm3d_split1000"), "vlfm": ("hm3d_rec_raw1000", "hm3d_rec_zson1000")}; AUC = {}
    G = lambda pl, run, rule: np.array([float(r["sev_" + rule]) for r in res if r["planner"] == pl and r["run"] == run and r["sev_" + rule] != ""])
    for pl, (a, b) in P.items():
        for rule in RULES:
            x, y = G(pl, a, rule), G(pl, b, rule); u = st.mannwhitneyu(x, y, alternative="greater"); AUC[pl, rule] = (len(x), len(y), np.median(x), np.median(y), u.statistic / (len(x) * len(y)), u.pvalue)
            rep.append(f"{pl:8s} {rule:11s} n={len(x)}/{len(y)} median={np.median(x):.3f}/{np.median(y):.3f} AUC={AUC[pl, rule][4]:.3f} p={u.pvalue:.4f}")
        rep.append("")
        for rule in RULES[1:]:
            D = [r for r in res if r["planner"] == pl and r["sev_decision"] != "" and r["sev_" + rule] != ""]
            s = st.spearmanr([float(r["sev_decision"]) for r in D], [float(r["sev_" + rule]) for r in D])
            rep.append(f"{pl:8s} spearman(decision, {rule:11s}) rho={s.statistic:.3f} p={s.pvalue:.4f} n={len(D)}")
        rep.append("")
    rep.append("skipped items per rule (item severity missing):")
    for rule in RULES: rep.append(f"  {rule}: {len(skip[rule])} " + "; ".join(f"{r} ep{e} ({'; '.join(n)})" for r, e, n in skip[rule][:40]))
    rep.append("unreadable frames dropped: " + "; ".join(BAD))
    rep.append(f"dropped before scoring (items and refs): {len(dropped)}: " + "; ".join(dropped))
    rep.append(f"window length (frames) over {len(wl)} items: min {min(wl)} median {np.median(wl):.0f} mean {np.mean(wl):.1f} max {max(wl)}")
    rep.append("notes: " + str(dict(zip(*np.unique([r["notes"] for r in res], return_counts=True)))))
    open(OUT + "tests.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
    C = {"raw": "#2a78d6", "gate": "#eb6834"}; rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, len(RULES), figsize=(3.3 * len(RULES), 8), sharey=True)
    for row, (pl, (a, b)) in zip(axs, P.items()):
        for ax, rule in zip(row, RULES):
            for x, (arm, run) in enumerate((("raw", a), ("gate", b))):
                v = G(pl, run, rule)
                ax.violinplot(v, [x], widths=0.7, showextrema=False)["bodies"][0].set(facecolor=C[arm], alpha=0.18, edgecolor="none")
                ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=C[arm], edgecolor="white", linewidth=0.6, zorder=3)
                ax.hlines(np.median(v), x - 0.25, x + 0.25, color="#333", lw=2, zorder=4)
                ax.text(x, 1.04, f"n={len(v)} median {np.median(v):.2f}", ha="center", fontsize=7, color="#555")
            n1, n2, _, _, auc, p = AUC[pl, rule]
            ax.set(xticks=[0, 1], xticklabels=["Baseline", "+ Verifier"], ylim=(-0.05, 1.12), title=f"{pl.replace('apexnav', 'ApexNav').replace('vlfm', 'VLFM')}  {rule}\nAUC {auc:.2f}  p {p:.3f}")
            ax.title.set_fontsize(9); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
        row[0].set_ylabel("FP severity (SigLIP)\n1 = least goal-like")
    fig.tight_layout(); fig.savefig(OUT + "sev_by_frame.png", dpi=130)


if __name__ == "__main__":
    main()
