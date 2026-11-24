"""Instance-level, human-aligned severity of eye-labelled false stops (ledger FS-perc): how much does the object the agent
committed to look like the GOAL (image-text, SigLIP / OpenCLIP, calibrated on real goal instances) vs like what it IS (DreamSim to THINGS images)?

  python -m nav.eval.fp_dreamsim items     items + crops + contact sheets + THINGS references -> OUT/
  python -m nav.eval.fp_dreamsim score     DreamSim / SigLIP / CLIP scores + pre-registered tests -> OUT/
(venv with torch cpu, dreamsim, transformers, open_clip_torch, scipy, requests; CPU only.)
Crop = bbox of the committed object's mask at the decision step (raw arms: planner mask / VLFM YOLOv7 box from
outputs/fp_severity/consensus/all_scores.tsv; gate arms: last ACCEPTED gate request mask.png on input.jpg), padded 10 %, square, clipped to the image.
Real-goal reference crops = posctrl rows of all_scores.tsv with a box, cropped the same way. Items without a crop are listed, not imputed."""
import csv, glob, io, json, os, re, sys, zipfile, collections
import numpy as np
from PIL import Image, ImageDraw
from nav.eval import fp_severity as fs
from nav.eval.fp_severity import ROOT, rows, record, decision_frame, accepted

OUT = ROOT + "/outputs/fp_severity/dreamsim/"
LAB = ROOT + "/outputs/apexnav_online/labels/"
ALL = ROOT + "/outputs/fp_severity/consensus/all_scores.tsv"
fs.RUNS["hm3d_split1000"] = (fs.A + "hm3d_split1000/gate", "apexnav", "hm3d", "gate")
ARMS = ["hm3d_raw1000", "hm3d_split1000", "hm3d_rec_raw1000", "hm3d_rec_zson1000"]
GOAL_CONCEPT = {"couch": "couch", "chair": "chair", "bed": "bed", "toilet": "toilet", "tv": "television", "potted plant": "plant"}
# selected names missing from things/mapping.tsv: head noun / first noun as there; None = no object concept
EXTRA = {"wooden chair": "chair", "blue futon sofa": "couch", "wooden frame / mesh hole": "wall", "door beside a toy": "door", "plain wall": "wall",
         "chair back": "chair", "grey stone object": None, "tire": None, "Victorian settee": "couch", "window daybed / armchair": "sofa bed",
         "glass door with a cushion": "door", "leopard armchairs": "chair", "door / pillar": "door", "blank bathroom wall": "wall", "white chair / stool": "chair",
         "recliner armchair": "chair", "window wall (nothing accepted in view)": None, "framed pictures on a wall": "photograph", "armchair / blanket": "chair",
         "door / wall (nothing accepted in view)": None, "sideboard / lamp": "cabinet", "cluster on a landing (nothing there)": None, "wall picture": "photograph",
         "wall frame": "photograph", "wall painting of flowers": "painting", "accent chair": "chair", "chaise / armchair": "chair", "white slipcovered sofa / bench": "couch",
         "maroon recliner / sofa-like seat": "recliner", "cushioned bench / fireplace": "bench", "gym exercise seat": "bench", "gym seat": "bench", "window pane": "window",
         "sofa end": "couch", "leopard armchair / chaise": "chair", "wall panel / desk cluster": "wall", "washer / dryer": "washing machine", "wall light panel / mirror": None,
         "backless counter stools": "stool", "tufted leather armchair": "chair", "outdoor dog bed": None, "futon / sofa": "sofa bed", "white armchair / chaise": "chair",
         "bench by an entry door": "bench", "sofas / armchairs": "couch", "black mesh hole above a bench": "wall", "sheepskin daybed / sofa": "sofa bed",
         "floral armchair / chaise": "chair", "gym bench / exercise machine": "bench", "leopard chaise / armchair": "chair", "transom window": "window",
         "framed picture on a wall": "photograph", "white storage bench": "bench", "teal dog bed": None, "small red object by a kitchen counter": None,
         "exercise bench with a red seat": "bench", "gaming / gym seat": "bench"}
ANCHORS = {"armchair for couch": ("couch", lambda t: "armchair" in t), "trash can for toilet": ("toilet", lambda t: "trash can" in t),
           "mirror for tv": ("tv", lambda t: "mirror" in t), "weight bench for chair": ("chair", lambda t: re.search(r"weight bench|exercise bench|gym bench", t) is not None)}


def true_names():
    """(run, ep) -> (failure_selected `selected`, goal), old run names mapped to the merged ones via run_merge."""
    F = {(r["run"], int(r["ep"])): (r["selected"], r["goal"]) for r in rows(LAB + "failure_selected.tsv")}
    out = {}
    for r in rows(LAB + "run_merge_2026-11-19.tsv"):
        if (r["old_run"], int(r["episode"])) in F: out[r["new_run"], int(r["episode"])] = F[r["old_run"], int(r["episode"])]
    out.update({k: v for k, v in F.items() if k[0] == "hm3d_rec_zson1000"})
    return out


def concept_of(text, M):
    if ("obj", text) in M: c = M[("obj", text)]["concept"]
    elif text in EXTRA: c = EXTRA[text] or "UNMAPPED"
    else: raise KeyError("unmapped selected name: " + text)
    return c


def square_crop(img, box):
    """bbox padded 10 %, made square (side = longer bbox side * 1.2, <= min image side), shifted to stay inside the image."""
    x0, y0, x1, y1 = box; W, H = img.size
    s = min(max(x1 - x0, y1 - y0) * 1.2, W, H); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    l = min(max(cx - s / 2, 0), W - s); t = min(max(cy - s / 2, 0), H - s)
    return img.crop((round(l), round(t), round(l + s), round(t + s)))


def erase_outline(img):
    """VLFM gate input.jpg has the planner's green mask contour baked in: inpaint the green pixels (+ a 2 px halo)."""
    import cv2
    a = np.asarray(img); g = (a[..., 1].astype(int) - np.maximum(a[..., 0], a[..., 2]) > 50).astype(np.uint8)
    g = cv2.dilate(g, np.ones((5, 5), np.uint8))
    return Image.fromarray(cv2.inpaint(a, g, 5, cv2.INPAINT_TELEA))


def mask_box(mk):
    ys, xs = np.where(np.asarray(mk) > 0)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else None


def things_download(concepts):
    """Extract only the needed concept folders of the 5 GB images_THINGS.zip by HTTP range reads (password in password_images.txt)."""
    import requests
    d = OUT + "download/things_images/"
    todo = [c for c in concepts if not glob.glob(d + c.replace(" ", "_") + "/*.jpg")]
    if not todo: return
    URL = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/670d6f7dbce84f4a7bb9371b?action=download"

    import time
    from concurrent.futures import ThreadPoolExecutor
    sess = requests.Session(); sig = [0, None]

    def url():   # the signed storage URL lives ~60 s: resolve the redirect once per 30 s
        if time.time() - sig[0] > 30:
            sig[1] = requests.get(URL, allow_redirects=False).headers["Location"]; sig[0] = time.time()
        return sig[1]

    class R(io.RawIOBase):
        n = int(requests.head("https://osf.io/download/rdxy2/", allow_redirects=True).headers["Content-Length"]); p = 0
        def seekable(s): return True
        def readable(s): return True
        def tell(s): return s.p
        def seek(s, o, w=0): s.p = {0: o, 1: s.p + o, 2: s.n + o}[w]; return s.p
        def read(s, k=-1):
            k = s.n - s.p if k < 0 else min(k, s.n - s.p)
            if k <= 0: return b""
            r = sess.get(url(), headers={"Range": f"bytes={s.p}-{s.p + k - 1}"}); r.raise_for_status(); s.p += k; return r.content
    names = zipfile.ZipFile(R()).namelist(); pw = b"things4all"; jobs = []
    for c in todo:
        f = c.replace(" ", "_"); m = [n for n in names if n.startswith(f"object_images/{f}/") and n.endswith(".jpg")]
        assert m, "no THINGS folder for concept: " + c
        os.makedirs(d + f, exist_ok=True); jobs += [(f, n) for n in m]
        print("THINGS", c, len(m), "images", flush=True)

    def get(j):
        z = zipfile.ZipFile(R()); open(d + j[0] + "/" + os.path.basename(j[1]), "wb").write(z.read(j[1], pwd=pw))
    with ThreadPoolExecutor(16) as ex: list(ex.map(get, jobs))


def items():
    L = [r for r in rows(fs.LEDGER) if r["label"] == "FS-perc" and r["run"] in ARMS]
    A = {(r["raw_run"], int(r["episode"])): r for r in rows(ALL) if r["group"] != "posctrl"}
    TN = true_names(); M = {(r["kind"], r["text"]): r for r in rows(ROOT + "/outputs/fp_severity/things/mapping.tsv")}
    os.makedirs(OUT + "crops/ref", exist_ok=True); out, ref = [], []
    for r in L:
        run, ep = r["run"], int(r["episode"]); _, planner, _, arm = fs.RUNS[run]
        sel, goal = TN[run, ep]
        it = dict(planner=planner, arm=arm, run=run, episode=ep, goal=goal, true_text=sel, crop="", full_frame="", note="")
        it["true_concept"] = concept_of(it["true_text"], M) if it["true_text"] else "UNMAPPED"
        try:
            rec, d = record(run, ep)
            if arm == "gate":
                a = accepted(d); img = Image.open(a + "/input.jpg").convert("RGB")
                if planner == "vlfm": img = erase_outline(img)
                box = mask_box(Image.open(a + "/mask.png").convert("L").resize(img.size))
                it["full_frame"] = decision_frame(run, ep)[0]
            else:
                a = A.get((run, ep)); box = None
                if a is None: it["note"] = "not in consensus all_scores (no YOLO box recovered)"
                else:
                    img = Image.open(ROOT + "/" + a["frame_path"]).convert("RGB"); it["full_frame"] = ROOT + "/" + a["frame_path"]
                    box = [int(v) for v in a["box_xyxy"].split()] if a["box_xyxy"] else None
                    if box is None: it["note"] = "no box: " + a["box_source"]
            if box is None and not it["note"]: it["note"] = "empty mask"
        except FileNotFoundError as e: box, it["note"] = None, "no gate accept / frame: " + str(e)[:60]
        if box:
            it["crop"] = f"{OUT}crops/{planner}_{arm}/{run}_ep{ep}.jpg"; os.makedirs(os.path.dirname(it["crop"]), exist_ok=True)
            square_crop(img, box).save(it["crop"], quality=92)
        out.append(it)
    for r in rows(ALL):   # real goal instances, same crop rule
        if r["group"] != "posctrl" or not r["box_xyxy"]: continue
        img = Image.open(ROOT + "/" + r["frame_path"]).convert("RGB"); p = f"{OUT}crops/ref/{r['planner']}_{r['raw_run']}_ep{r['episode']}.jpg"
        square_crop(img, [int(v) for v in r["box_xyxy"].split()]).save(p, quality=92)
        ref.append(dict(planner=r["planner"], run=r["raw_run"], episode=r["episode"], goal=r["goal"], crop=p, full_frame=ROOT + "/" + r["frame_path"]))
    for name, data in (("items", out), ("ref_items", ref)):
        with open(OUT + name + ".tsv", "w") as f:
            w = csv.DictWriter(f, list(data[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(data)
    cs = {c for it in out for c in (it["true_concept"],) if c != "UNMAPPED"} | set(GOAL_CONCEPT.values())
    with open(OUT + "mapping.tsv", "w") as f:   # used true-name -> concept, with the source of each mapping
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(["true_text", "concept", "source"])
        for t in sorted({it["true_text"] for it in out if it["true_text"]}):
            w.writerow([t, concept_of(t, M), "things/mapping.tsv" if ("obj", t) in M else "fp_dreamsim.EXTRA (head/first noun; None = not an object)"])
    things_download(sorted(cs)); sheets(out)
    print(len(out), "items,", sum(1 for i in out if i["crop"]), "with crop;", len(ref), "reference crops")


def sheets(out):
    for key in sorted({(i["planner"], i["arm"]) for i in out}):
        it = [i for i in out if (i["planner"], i["arm"]) == key and i["crop"]]; C, W = 6, 200
        S = Image.new("RGB", (C * W, ((len(it) + C - 1) // C) * (W + 30)), "white"); dr = ImageDraw.Draw(S)
        for k, i in enumerate(it):
            x, y = W * (k % C), (W + 30) * (k // C)
            S.paste(Image.open(i["crop"]).resize((W, W)), (x, y))
            dr.text((x + 2, y + W + 1), f"goal={i['goal']}  ep{i['episode']}", fill="black"); dr.text((x + 2, y + W + 14), "true=" + i["true_text"][:30], fill="red")
        S.save(f"{OUT}sheet_{key[0]}_{key[1]}.jpg", quality=88)


def pct(v, ref):
    """fraction of real-goal reference values <= v"""
    return float((np.asarray(ref) <= v).mean())


def score():
    import torch, scipy.stats as st
    torch.set_num_threads(8)
    from dreamsim import dreamsim
    items_ = [i for i in rows(OUT + "items.tsv")]; ref = rows(OUT + "ref_items.tsv")
    dm, dp = dreamsim(pretrained=True, device="cpu", cache_dir=OUT + "download/dreamsim_weights")
    def emb(paths):
        with torch.no_grad(): e = torch.cat([dm.embed(dp(Image.open(p).convert("RGB"))) for p in paths])
        return torch.nn.functional.normalize(e, dim=-1)
    bank = {}
    def dist(crop, concept):   # mean DreamSim distance (1 - cos of the ensemble embedding) from crop to the concept's THINGS images
        if concept not in bank: bank[concept] = emb(sorted(glob.glob(OUT + "download/things_images/" + concept.replace(" ", "_") + "/*.jpg")))
        return float((1 - emb([crop]) @ bank[concept].T).mean())
    for r in ref: r["d_goal"] = dist(r["crop"], GOAL_CONCEPT[r["goal"]])
    for i in items_:
        if not i["crop"]: continue
        i["d_goal"] = dist(i["crop"], GOAL_CONCEPT[i["goal"]])
        if i["true_concept"] != "UNMAPPED": i["d_true"] = dist(i["crop"], i["true_concept"])
    # image-text: SigLIP (primary, loading as in fp_severity.score) and OpenCLIP ViT-L/14 laion2b
    from transformers import AutoModel, AutoProcessor
    sm, sp = AutoModel.from_pretrained(fs.MODEL).eval(), AutoProcessor.from_pretrained(fs.MODEL)
    import open_clip
    cm, _, cp = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k"); cm.eval(); ct = open_clip.get_tokenizer("ViT-L-14")
    def text_scores(paths, goals):
        s, c = [], []
        with torch.no_grad():
            for p, g in zip(paths, goals):
                t = f"a photo of a {g}"; img = Image.open(p).convert("RGB")
                s.append(sm(**sp(text=[t], images=img, padding="max_length", return_tensors="pt")).logits_per_image[0, 0].item())
                a = torch.nn.functional.normalize(cm.encode_image(cp(img)[None]), dim=-1); b = torch.nn.functional.normalize(cm.encode_text(ct([t])), dim=-1)
                c.append((a @ b.T).item())
        return s, c
    for key, col in (("crop", "crop"), ("full", "full_frame")):
        for D in (ref, [i for i in items_ if i[col]]):
            s, c = text_scores([d[col] for d in D], [d["goal"] for d in D])
            for d, a, b in zip(D, s, c): d["siglip_" + key], d["clip_" + key] = a, b
    refn = collections.Counter(r["goal"] for r in ref); print("real-goal reference crops per goal:", dict(refn), "WARN n<8:", [g for g, n in refn.items() if n < 8])
    for i in items_:
        for k in ("crop", "full"):
            for m in ("siglip", "clip"):
                if i.get(f"{m}_{k}") is not None:
                    i[f"sev_{m}_{k}"] = 1 - pct(i[f"{m}_{k}"], [float(r[f"{m}_{k}"]) for r in ref if r["goal"] == i["goal"]])   # 1 - goal_pct
        if "d_goal" in i:
            i["d_goal_pct"] = pct(i["d_goal"], [r["d_goal"] for r in ref if r["goal"] == i["goal"]])
            if "d_true" in i: i["typ_margin"] = i["d_goal"] - i["d_true"]
    cols = list(dict.fromkeys(k for i in items_ for k in i))
    for name, D in (("scores.tsv", items_), ("ref_scores.tsv", ref)):
        with open(OUT + name, "w") as f:
            w = csv.DictWriter(f, list(dict.fromkeys(k for i in D for k in i)), delimiter="\t", lineterminator="\n", restval=""); w.writeheader(); w.writerows(D)
    G = lambda run, m: [float(i[m]) for i in items_ if i["run"] == run and i.get(m) not in (None, "")]
    rep = []
    def test(name, a, b, m):
        x, y = G(a, m), G(b, m); u = st.mannwhitneyu(x, y, alternative="greater")
        rep.append(f"{name:42s} {m:15s} n={len(x)}/{len(y)} mean={np.mean(x):.3f}/{np.mean(y):.3f} AUC={u.statistic / (len(x) * len(y)):.3f} p={u.pvalue:.4f}")
    ms = ["sev_siglip_crop", "typ_margin", "d_goal_pct", "sev_clip_crop", "sev_siglip_full", "sev_clip_full"]
    for m in ms: test("T1 VLFM raw FS > VLFM gate FS", "hm3d_rec_raw1000", "hm3d_rec_zson1000", m)
    for m in ms: test("T2 ApexNav gate FS > VLFM gate FS", "hm3d_split1000", "hm3d_rec_zson1000", m)
    rep.append("")
    for run in ARMS:
        D = [i for i in items_ if i["run"] == run]
        rep.append(f"{run}: FS-perc n={len(D)}, with crop {sum(1 for i in D if i['crop'])}, with true concept+crop {sum(1 for i in D if 'typ_margin' in i)}; no crop: " +
                   "; ".join(f"ep{i['episode']} ({i['note']})" for i in D if not i["crop"]))
        rep.append("   unmapped true name (no DreamSim true score): " + "; ".join(f"ep{i['episode']} '{i['true_text']}'" for i in D if i["crop"] and "typ_margin" not in i))
    rep.append("")
    for name, (g, f) in ANCHORS.items():
        D = [i for i in items_ if i["goal"] == g and f(i["true_text"].lower()) and "typ_margin" in i]
        rep.append(f"anchor {name}: n={len(D)} (all arms pooled) " + "  ".join(f"{m}={np.mean([float(i[m]) for i in D if m in i]):.3f}" for m in ["typ_margin", "d_goal_pct", "sev_siglip_crop"]) if D else f"anchor {name}: n=0")
    open(OUT + "tests.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))


if __name__ == "__main__":
    {"items": items, "score": score}[sys.argv[1]]()
