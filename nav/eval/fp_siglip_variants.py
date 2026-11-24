"""Reference-free SigLIP severity variants for false stops (nothing fitted to our runs). CPU only, reads cached image embeddings.
  CUDA_VISIBLE_DEVICES="" sam3d-env sam3d python -m nav.eval.fp_siglip_variants   -> outputs/fp_severity/variants/{scores,tests}.tsv, variants.png, extremes.jpg
A 1-p(goal) softmax | B max wrong logit - max goal-synonym logit | C normalised rank of best goal synonym | D 1-sigmoid(goal logit) | F THINGS image prototypes | P percentile score (frames/scores.tsv).
Severity of a frame = -goal-likeness; multi-frame rules aggregate goal-likeness by the rule's mean/max as in fp_frames."""
import csv, glob, os, re
import numpy as np
from nav.eval import fp_dreamsim as dd, fp_frames as ff, fp_severity as fs

ROOT = fs.ROOT + "/outputs/fp_severity/"; OUT = ROOT + "variants/"; VOC = ROOT + "vocab/"
THINGS = ROOT + "dreamsim/download/things_images/"
RULES = ["decision", "final", "win_max", "all30_mean"]
PL = {"apexnav": ("hm3d_raw1000", "hm3d_split1000"), "vlfm": ("hm3d_rec_raw1000", "hm3d_rec_zson1000")}
T1 = ["a photo of a {}."]
T5 = ["a photo of a {}.", "a photo of a {} in a room.", "an indoor photo with a {}.", "a {} in a house.", "a photo of the {}."]
COCO = """person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush""".split("|")
SYN = {"couch": ["couch", "sofa"], "chair": ["chair", "office chair", "armchair", "sofa chair", "folded chair"], "bed": ["bed", "bunk bed"], "toilet": ["toilet"], "tv": ["tv", "television"], "potted plant": ["plant", "potted plant"]}


def scannet():
    s = open(VOC + "scannet200_constants.py").read()
    L = eval("[" + re.search(r"CLASS_LABELS_200 = \((.*?)\)", s, re.S).group(1) + "]")
    return [x for x in L if x not in ("wall", "floor", "ceiling")]


def load_model():
    import torch
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(4)
    return AutoModel.from_pretrained(fs.MODEL).eval(), AutoProcessor.from_pretrained(fs.MODEL)


def text_emb(m, pr, words, templates):
    """Template-averaged, normalised text embeddings; per-(template, word) embeddings cached in OUT/text_cache.npz."""
    import torch
    f = OUT + "text_cache.npz"; C = dict(np.load(f)) if os.path.exists(f) else {}
    miss = [(t, w) for t in templates for w in words if f"{fs.MODEL}|{t}|{w}" not in C]
    with torch.no_grad():
        for i in range(0, len(miss), 64):
            b = miss[i:i + 64]; x = pr(text=[t.format(w) for t, w in b], padding="max_length", return_tensors="pt").to(m.device)
            e = m.get_text_features(**x); e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
            C.update({f"{fs.MODEL}|{t}|{w}": v for (t, w), v in zip(b, e)})
    if miss: np.savez(f, **C)
    e = np.mean([[C[f"{fs.MODEL}|{t}|{w}"] for w in words] for t in templates], 0); return e / np.linalg.norm(e, axis=1, keepdims=True)


def prototypes(m, pr):
    import torch
    from PIL import Image
    f = OUT + "protos.npz"
    if os.path.exists(f): z = np.load(f); return dict(zip(z["names"], z["P"]))
    names, P = [], []
    for c in sorted(os.listdir(THINGS)):
        ps = sorted(glob.glob(THINGS + c + "/*")); E = []
        for i in range(0, len(ps), 8):
            with torch.no_grad():
                e = m.get_image_features(**pr(images=[Image.open(p).convert("RGB") for p in ps[i:i + 8]], return_tensors="pt"))
            E.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        v = np.concatenate(E).mean(0); names.append(c); P.append(v / np.linalg.norm(v))
    np.savez(f, names=np.array(names), P=np.stack(P)); return dict(zip(names, P))


def main():
    import scipy.stats as st
    os.makedirs(OUT, exist_ok=True)
    z = np.load(ROOT + "frames/img_emb.npz"); idx = {p: i for i, p in enumerate(z["paths"])}; E = z["emb"].astype(np.float32); S = float(np.exp(z["logit_scale"])); B = float(z["logit_bias"])
    voc = {"scannet200": scannet(), "coco": COCO}; goals = list(SYN); syn = {}   # synonyms present in a vocabulary; the goal word is added if absent
    for v, W in voc.items():
        for g in goals:
            s = [w for w in SYN[g] if w in W] or [g]
            if g not in s: s.append(g)
            syn[v, g] = s
    for (v, g), ss in syn.items(): voc[v] = voc[v] + [w for w in ss if w not in voc[v]]
    open(OUT + "synonyms.txt", "w").write("\n".join(f"{v}\t{g}\t{s}" for (v, g), s in syn.items()) + "\n")
    m, pr = load_model()
    TX = {}   # (vocab, tmpl) -> (words, emb); goal words embedded separately for D
    for v, W in voc.items():
        for tn, T in (("T1", T1), ("T5", T5)):
            TX[v, tn] = text_emb(m, pr, voc[v], T)
    GW = {tn: dict(zip(goals, text_emb(m, pr, goals, T))) for tn, T in (("T1", T1), ("T5", T5))}
    proto = prototypes(m, pr); del m
    GC = dd.GOAL_CONCEPT; PN = sorted(proto); PM = np.stack([proto[c] for c in PN])

    items = ff.read(ff.OUT + "scores.tsv"); refs = ff.read(dd.OUT + "ref_items.tsv")
    def has(r):
        try: fs.record(r["run"], int(r["episode"])); return True
        except FileNotFoundError: return False
    refs = [r for r in refs if has(r)]

    def sev_frames(e, goal):
        out = {}
        for v in voc:
            for tn in ("T1", "T5"):
                N = len(voc[v]); lg = S * e @ TX[v, tn].T + B; gi = [voc[v].index(w) for w in syn[v, goal]]
                gl = lg[:, gi]; wrong = np.delete(lg, gi, axis=1)
                pr_ = np.exp(lg - lg.max(1, keepdims=True)); pr_ /= pr_.sum(1, keepdims=True); pg = pr_[:, gi].sum(1)
                out[f"A_{v}_{tn}"] = 1 - pg
                out[f"B_{v}_{tn}"] = wrong.max(1) - gl.max(1)
                if v == "scannet200" and tn == "T5": out["C_scannet200_T5"] = (lg > gl.max(1, keepdims=True)).sum(1) / (N - 1)
                out["_top_" + v + tn] = (lg, pg)
        for tn in ("T1", "T5"): out[f"D_{tn}"] = 1 - 1 / (1 + np.exp(-(S * e @ GW[tn][goal] + B)))
        cs = e @ PM.T; gi = PN.index(GC[goal]); out["F"] = np.delete(cs, gi, 1).max(1) - cs[:, gi]
        return out
    VAR = [k for k in sev_frames(E[:1], "bed") if not k.startswith("_")] + ["P"]

    def item_vals(r):
        run, ep, goal = r["run"], int(r["episode"]), r["goal"]
        stored, dec, _ = ff.frames(run, ep); w, _ = ff.window(run, ep)
        paths = sorted({p for _, p in stored} | {p for _, p in w} | ({dec[0]} if dec else set()))
        sv = sev_frames(E[[idx[p] for p in paths]], goal); pos = {p: i for i, p in enumerate(paths)}
        res = {}
        for k, s in sv.items():
            if k.startswith("_"): continue
            g = -s; f = lambda ps, a: -a(g[[pos[p] for p in ps]])
            res[k] = {"decision": -g[pos[dec[0]]] if dec else None, "final": -g[pos[stored[-1][1]]], "win_max": f([p for _, p in w], np.max), "all30_mean": f([p for _, p in stored], np.mean)}
        return res, (dec[0] if dec else None), paths, sv, pos

    rows, REF = [], {}
    for kind, L in (("item", items), ("ref", refs)):
        for r in L:
            v, dp, paths, sv, pos = item_vals(r)
            if kind == "item":
                v["P"] = {rl: (float(r["sev_" + rl]) if r["sev_" + rl] != "" else None) for rl in RULES}
                if dp:
                    lg, pg = sv["_top_scannet200T5"]; lab = voc["scannet200"]; l = lg[pos[dp]]; top = [lab[i] for i in np.argsort(-l)[:3]]
                    r["_dec"] = dict(path=dp, top=top, pg=float(pg[pos[dp]]), sev=float(v["A_scannet200_T5"]["decision"]))
                rows.append((r, v))
            else: REF[r["run"], r["episode"]] = v
    # scores.tsv
    with open(OUT + "scores.tsv", "w") as f:
        cols = ["planner", "arm", "run", "episode", "goal"] + [f"{k}__{rl}" for k in VAR for rl in RULES]
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(cols)
        for r, v in rows + [(dict(planner=r["planner"], arm="ref", run=r["run"], episode=r["episode"], goal=r["goal"]), REF[r["run"], r["episode"]]) for r in refs if (r["run"], r["episode"]) in REF]:
            w.writerow([r["planner"], r["arm"], r["run"], r["episode"], r["goal"]] + ["" if v.get(k, {}).get(rl) is None else f"{v[k][rl]:.6g}" for k in VAR for rl in RULES])
    # tests
    def vals(pl, run, k, rl): return np.array([v[k][rl] for r, v in rows if r["planner"] == pl and r["run"] == run and v[k][rl] is not None])
    refv = lambda k, rl: np.array([v[k][rl] for v in REF.values() if v[k][rl] is not None])
    T = []; AU = {}
    for k in VAR:
        for rl in RULES:
            rv = refv(k, rl) if k != "P" else None
            for pl, (a, b) in PL.items():
                x, y = vals(pl, a, k, rl), vals(pl, b, k, rl); u = st.mannwhitneyu(x, y, alternative="greater"); allv = np.concatenate([x, y])
                row = dict(variant=k, vocab=k.split("_")[1] if k[0] in "AB" and "_" in k else ("scannet200" if k[0] == "C" else ""), template=k.split("_")[-1] if k[0] in "ABCD" else "", rule=rl, planner=pl, n_base=len(x), n_ver=len(y),
                           med_base=np.median(x), med_ver=np.median(y), auc=u.statistic / (len(x) * len(y)), p=u.pvalue)
                if rv is not None and k != "P":
                    uv = st.mannwhitneyu(allv, rv, alternative="greater"); row.update(auc_valid=uv.statistic / (len(allv) * len(rv)), p_valid=uv.pvalue)
                else: row.update(auc_valid="", p_valid="")   # P is a percentile of the references themselves
                row["sat"] = np.mean((allv == allv.min()) | (allv == allv.max())); row["ties"] = 1 - len(np.unique(allv)) / len(allv)
                if rl == "decision":
                    pp = [(v[k][rl], v["P"][rl]) for r, v in rows if r["planner"] == pl and v[k][rl] is not None and v["P"][rl] is not None]
                    row["rho_P"] = st.spearmanr(*zip(*pp)).statistic
                T.append(row); AU[k, rl, pl] = row
    cols = ["variant", "vocab", "template", "rule", "planner", "n_base", "n_ver", "med_base", "med_ver", "auc", "p", "auc_valid", "p_valid", "sat", "ties", "rho_P"]
    with open(OUT + "tests.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(cols)
        for r in T: w.writerow([("%.4g" % r.get(c)) if isinstance(r.get(c), (float, np.floating)) else r.get(c, "") for c in cols])
    plot(rows, AU); extremes(rows)
    for k in VAR: print(k, "mean AUC decision %.3f" % np.mean([AU[k, "decision", pl]["auc"] for pl in PL]))


def plot(rows, AU):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"raw": "#2a78d6", "gate": "#eb6834"}; rng = np.random.default_rng(0)
    P = ["A_scannet200_T5", "B_scannet200_T5", "C_scannet200_T5", "D_T5", "F", "P"]
    fig, axs = plt.subplots(2, len(P), figsize=(3.2 * len(P), 7.5))
    for row, (pl, (a, b)) in zip(axs, PL.items()):
        for ax, k in zip(row, P):
            for x, (arm, run) in enumerate((("raw", a), ("gate", b))):
                v = np.array([vv[k]["decision"] for r, vv in rows if r["run"] == run and vv[k]["decision"] is not None])
                ax.violinplot(v, [x], widths=0.7, showextrema=False)["bodies"][0].set(facecolor=C[arm], alpha=0.18, edgecolor="none")
                ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=C[arm], edgecolor="white", linewidth=0.6, zorder=3)
                ax.hlines(np.median(v), x - 0.25, x + 0.25, color="#333", lw=2, zorder=4)
            t = AU[k, "decision", pl]
            ax.set(xticks=[0, 1], xticklabels=["Baseline", "+ Verifier"], title=f"{pl.replace('apexnav', 'ApexNav').replace('vlfm', 'VLFM')}  {k}\nAUC {t['auc']:.2f}  p {t['p']:.3f}")
            ax.title.set_fontsize(9); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
    fig.suptitle("False-stop severity, decision frame (higher = more obviously wrong)"); fig.tight_layout(); fig.savefig(OUT + "variants.png", dpi=120)


def extremes(rows):
    from PIL import Image, ImageDraw
    D = sorted([r for r, _ in rows if "_dec" in r], key=lambda r: r["_dec"]["sev"]); sel = D[-6:][::-1] + D[:6]
    W, H, TH = 400, 300, 62; sheet = Image.new("RGB", (6 * W, 2 * (H + TH)), "white"); d = ImageDraw.Draw(sheet)
    for i, r in enumerate(sel):
        x, y = (i % 6) * W, (i // 6) * (H + TH); q = r["_dec"]
        sheet.paste(Image.open(q["path"]).convert("RGB").resize((W, H)), (x, y))
        d.text((x + 4, y + H + 2), f"{'HIGH' if i < 6 else 'LOW'} sev {q['sev']:.3f}  goal={r['goal']}  {r['planner']}/{r['arm']} ep{r['episode']}", fill="black")
        d.text((x + 4, y + H + 18), "top3: " + ", ".join(q["top"]), fill="black"); d.text((x + 4, y + H + 34), f"p(goal)={q['pg']:.4f}", fill="black")
    sheet.save(OUT + "extremes.jpg", quality=88)


if __name__ == "__main__":
    main()
