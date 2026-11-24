"""False-stop severity from images: how obviously wrong is the object a planner stopped at?

  python3 -m nav.eval.fp_severity sheets            one contact sheet per eye-labelled false stop (ledger label FS-perc)
  sam3d-env sam3d python -m nav.eval.fp_severity score [--gpu N]   SigLIP scores -> OUT/scores.tsv

LABELS.tsv (by eye from the sheets, never from the semantic sensor): run, episode, ..., true_name (other columns unused).
Scored image = the full decision frame (decision_frame): raw arms the planner's commit to the object it stopped at,
gate arms the view the last accept was judged on (clean rgb, no overlay).
Scores per decision frame (SigLIP, prompt "a photo of a {}"); only rows still labelled FS-perc in the ledger:
  margin   = s(frame, true_name) - s(frame, target): high = the object plainly is what it is (an obvious mistake),
             <= 0 = it really looks like the target
  p_target = softmax over {target, true_name} of the SigLIP logits
  text_dist = 1 - cos(text(target), text(true_name)): category distance without the image"""
import argparse, csv, glob, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LEDGER = ROOT + "/outputs/apexnav_online/labels/ledger.tsv"
OUT = ROOT + "/outputs/fp_severity/"
A, V = ROOT + "/planners/ApexNav/videos/", ROOT + "/planners/VLFM/outputs/"
# run -> (episode root, planner, dataset, arm)
RUNS = {"hm3d_raw1000": (A + "hm3d_raw1000/baseline", "apexnav", "hm3d", "raw"),
        "ovon_raw200": (A + "ovon_raw200/baseline", "apexnav", "ovon", "raw"),
        "ovon_split200": (A + "ovon_split200/gate", "apexnav", "ovon", "gate"),
        "hm3d_rec_raw1000": (V + "hm3d_rec_raw1000/baseline", "vlfm", "hm3d", "raw"),
        "hm3d_rec_zson1000": (V + "hm3d_rec_zson1000/gate", "vlfm", "hm3d", "gate")}
MODEL = "google/siglip-so400m-patch14-384"


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def record(run, ep):
    """(record dir, episode dir) of one episode; the record dir holds rgb/, meta.json."""
    d = f"{RUNS[run][0]}/epi{ep}"
    rec = sorted(glob.glob(d + "/records/*/*/rgb"))
    if not rec: raise FileNotFoundError(f"no frames: {run} ep{ep}")
    return os.path.dirname(rec[-1]) + "/", d


def accepted(d):
    """Last gate-accepted view (input.jpg, mask.png) of an episode, as in make_evidence.py mont."""
    acc = []
    for r in map(json.loads, open(d + "/verification/decisions.jsonl")):
        if r.get("state") != 1: continue
        fs = [(int(p.rsplit("_s", 1)[1]), p) for p in glob.glob(f"{d}/verification/epi*_*/c{r['cluster']['id']}_s*")]
        fs = [x for x in fs if x[0] <= r["step"]]
        if fs: acc.append((r["step"], max(fs)[1]))
    if not acc: raise FileNotFoundError(f"gate accepted nothing: {d}")
    return max(acc)[1]


def decision_frame(run, ep):
    """(image path, rgb step) of the frame the stop decision was made on. Frame index of steps.jsonl step S (verified by mask overlay):
    ApexNav rgb/NNNNNN.jpg = step N (detection masks/N_k.png are on rgb N); VLFM rgb/NNNNNN.jpg = step N+1 (frame i = step i+1's observation).
    raw: first step of the last stretch with final_state != EXPLORE (the planner's commit to the object it stopped at).
    gate: the last accept: its server evidence frame, or for an approach accept its last counted view (< 0 = a bank view)."""
    rec, d = record(run, ep)
    if RUNS[run][3] == "raw":
        fs = [json.loads(l)["final_state"] for l in open(rec + "steps.jsonl")]
        t = len(fs)
        while t > 0 and fs[t - 1] != 0: t -= 1
        n = t + (RUNS[run][1] == "apexnav")   # list index t = steps.jsonl step t + 1; ApexNav rgb = step, VLFM rgb = step - 1
    else:
        a = [r for r in map(json.loads, open(d + "/verification/decisions.jsonl")) if r.get("state") == 1][-1]
        if a.get("approach") is None:
            n = a["evidence"]["step"]
        else:
            v = a["approach"]["views"][-1]
            ap = max(glob.glob(f"{d}/verification/approach/epi*_c{a['cluster']['id']}_s*"), key=lambda p: int(p.rsplit("_s", 1)[1]))
            if v >= 0:
                return glob.glob(f"{ap}/frames/f{v:03d}_a*.jpg")[0], None
            n = [e for e in map(json.loads, open(ap + "/approach.jsonl")) if e.get("event") == "seed" and e["step"] == v][-1]["bank_step"]
    f = f"{rec}rgb/{n:06d}.jpg"
    if not os.path.exists(f): raise FileNotFoundError(f"decision frame not stored: {f}")
    return f, n


def false_stops():
    return [r for r in rows(LEDGER) if r["label"] == "FS-perc" and r["run"] in RUNS]


def sheets():
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    os.makedirs(OUT + "sheets", exist_ok=True)
    jobs = []
    for r in false_stops():
        run, ep = r["run"], r["episode"]
        try: rec, d = record(run, ep)
        except FileNotFoundError as e: print(e); continue
        target = json.load(open(rec + "meta.json"))["target_label"]
        fr = sorted(os.listdir(rec + "rgb"))[-8:]
        W, H = 320, 240
        gate = RUNS[run][3] == "gate"
        S = Image.new("RGB", (4 * W, (3 if gate else 2) * (H + 18) + 22), "white"); dr = ImageDraw.Draw(S)
        dr.text((3, 4), f"{run} ep{ep}  target={target}  (last 8 frames, final = bottom right)", fill="black")
        for k, f in enumerate(fr):
            x, y = W * (k % 4), 22 + (H + 18) * (k // 4)
            dr.text((x + 3, y), f, fill="blue"); S.paste(Image.open(rec + "rgb/" + f).convert("RGB").resize((W, H)), (x, y + 16))
        if gate:
            a = accepted(d)
            im = Image.open(a + "/input.jpg").convert("RGB"); mk = Image.open(a + "/mask.png").convert("L").resize(im.size)
            edge = np.asarray(mk.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5))) > 0
            arr = np.asarray(im).copy(); arr[edge] = [255, 0, 0]
            y = 22 + 2 * (H + 18)
            dr.text((3, y), "accept: last gate-ACCEPTED view (red = accepted mask)", fill="red")
            S.paste(Image.fromarray(arr).resize((W, H)), (0, y + 16))
        S.save(f"{OUT}sheets/{run}__ep{ep}.jpg", quality=88)
        jobs.append(dict(run=run, episode=ep, target=target, arm=RUNS[run][3], ledger_note=r["note"]))
    with open(OUT + "jobs.tsv", "w") as fo:
        w = csv.DictWriter(fo, list(jobs[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(jobs)
    print(len(jobs), "sheets in", OUT + "sheets")



def score(gpu):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    model = AutoModel.from_pretrained(MODEL).to(dev).eval(); proc = AutoProcessor.from_pretrained(MODEL)
    jobs = {(j["run"], j["episode"]): j for j in rows(OUT + "jobs.tsv")}
    fs = {(r["run"], r["episode"]) for r in false_stops()}
    out = []
    with torch.no_grad():
        for r in rows(OUT + "labels.tsv"):
            if (r["run"], r["episode"]) not in fs: print("not FS-perc in ledger, skipped:", r["run"], r["episode"]); continue
            try: img = decision_frame(r["run"], r["episode"])[0]
            except FileNotFoundError as e: print("no decision frame, skipped:", r["run"], r["episode"], e); continue
            j = jobs[(r["run"], r["episode"])]; t, n = j["target"], r["true_name"]
            x = proc(text=[f"a photo of a {t}", f"a photo of a {n}"], images=Image.open(img).convert("RGB"),
                     padding="max_length", return_tensors="pt").to(dev)
            o = model(**x); lg, txt = o.logits_per_image[0], o.text_embeds
            _, planner, dataset, arm = RUNS[r["run"]]
            out.append(dict(run=r["run"], episode=r["episode"], planner=planner, dataset=dataset, arm=arm, target=t, true_name=n,
                            s_target=round(lg[0].item(), 3), s_true=round(lg[1].item(), 3), margin=round((lg[1] - lg[0]).item(), 3),
                            p_target=round(lg.softmax(0)[0].item(), 3), text_dist=round(1 - (txt[0] @ txt[1]).item(), 3)))
    with open(OUT + "scores.tsv", "w") as fo:
        w = csv.DictWriter(fo, list(out[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(out)
    print(len(out), "scored ->", OUT + "scores.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sheets", "score"])
    ap.add_argument("--gpu", type=int, default=-1); ap.add_argument("--out", help="output folder (default outputs/fp_severity)")
    a = ap.parse_args(); OUT = a.out.rstrip("/") + "/" if a.out else OUT
    {"sheets": sheets, "score": lambda: score(a.gpu)}[a.cmd]()
