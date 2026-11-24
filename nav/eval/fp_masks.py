"""Mask of the object the agent stopped at / accepted, on (or nearest to) the decision frame of every blind-key item.

  python -m nav.eval.fp_masks index     outputs/fp_severity/masks/{index.tsv,png/<id>.png,coverage.tsv,missing.tsv}
  python -m nav.eval.fp_masks sheets    outputs/fp_severity/masks/check_<k>.jpg (24 random items seed 0 + all frame_offset != 0)
(CPU only; sam3d env for numpy/PIL/cv2.)
Sources (frame_offset = decision rgb index - rgb index the mask belongs to; rgb index = steps.jsonl step for ApexNav, step - 1 for VLFM; verified by mask overlay):
  apexnav_det_at_decision    ApexNav raw/ref: planner_commit.matched_detection_index at the decision step -> masks/<step>_<k>.png
  apexnav_det_nearest_step   ApexNav raw/ref: no usable match at the decision step -> detection of a nearby step whose world_centroid is
                             within MATCH_M of the committed cluster centroid (offset != 0 possible)
  vlfm_annot_at_decision / vlfm_annot_nearest_step   VLFM raw/ref: no mask is stored, so the planner's green contour in rgb_annot (diff to
                             rgb) is filled; with several contours the one whose area best matches the committed detection's semantic pixel count
  gate_accept                gate arms: the last accepted request's mask.png (or approach frame mask) beside the frame it was judged on
  consensus_yolo             box only (outputs/fp_severity/consensus/all_scores.tsv) when no mask exists and that box is on the decision frame
Nothing is imputed: an item with no source is listed with source=none."""
import csv, glob, json, os, sys, collections
import numpy as np
from PIL import Image
import cv2
from nav.eval import fp_severity as fs
from nav.eval.fp_severity import ROOT, rows

fs.RUNS["hm3d_split1000"] = (fs.A + "hm3d_split1000/gate", "apexnav", "hm3d", "gate")
OUT = ROOT + "/outputs/fp_severity/masks/"
KEY = ROOT + "/outputs/fp_severity/blind_key.tsv"
ALL = ROOT + "/outputs/fp_severity/consensus/all_scores.tsv"
MATCH_M = 1.0      # max distance detection centroid <-> committed cluster centroid
MAX_OFF = 12       # steps searched around the decision step
ALIGN = 8.0        # max mean abs pixel diff between a gate view and the decision frame
COLS = ["id", "run", "episode", "kind", "arm", "goal", "decision_frame", "mask_path", "box_xyxy", "source", "frame_offset", "note"]


def mask_box(m):
    ys, xs = np.where(m > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def read_mask(path, size):
    m = Image.open(path).convert("L")
    if m.size != size: m = m.resize(size, Image.NEAREST)
    return np.asarray(m) > 0


def steps_of(rec):
    return {r["step"]: r for r in map(json.loads, open(rec + "steps.jsonl"))}


def green_components(rec, n):
    """Filled external contours of the planner's green outline in rgb_annot/n vs rgb/n (VLFM). [] if no annotated frame."""
    fa, fr = f"{rec}rgb_annot/{n:06d}.jpg", f"{rec}rgb/{n:06d}.jpg"
    if not (os.path.exists(fa) and os.path.exists(fr)): return None
    a, r = np.asarray(Image.open(fa).convert("RGB")).astype(int), np.asarray(Image.open(fr).convert("RGB")).astype(int)
    g = ((a[..., 1] - np.maximum(a[..., 0], a[..., 2]) > 80) & (np.abs(a - r).sum(-1) > 90)).astype(np.uint8)
    g = cv2.dilate(g, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 30: continue
        m = np.zeros(g.shape, np.uint8); cv2.drawContours(m, [c], -1, 1, -1); out.append(m.astype(bool))
    return out


def pick_component(comps, det, all_dets):
    """Component whose area best matches the semantic pixel count of detection `det` (VLFM masks are not stored)."""
    px = lambda d: sum(i["pixels"] for i in (d.get("semantic_instances") or []))
    if len(comps) == 1: return comps[0], ""
    P = px(det)
    if not P: return None, f"{len(comps)} contours, detection has no semantic pixels"
    cost = sorted((abs(np.log(max(c.sum(), 1) / P)), i) for i, c in enumerate(comps))
    note = f"{len(comps)} contours, area match |log ratio| {cost[0][0]:.2f}"
    if len(cost) > 1 and cost[1][0] - cost[0][0] < 0.15: note += " AMBIGUOUS"
    return comps[cost[0][1]], note


def raw_mask(planner, rec, n, size):
    """(mask bool HxW or None, source, offset, note) for a raw/ref item with decision rgb index n."""
    S = steps_of(rec); s = n + (planner != "apexnav")   # steps.jsonl step of rgb index n
    if s not in S: return None, "none", "", f"step {s} not in steps.jsonl"
    commit = S[s].get("planner_commit")
    if commit is None:   # first commit at/after s
        later = [t for t in sorted(S) if t > s and S[t].get("planner_commit")]
        commit = S[later[0]]["planner_commit"] if later else None
    far = MAX_OFF
    if commit is None:   # ApexNav SEARCH_SUSPICIOUS_OBJECT (expl_result 3) logs no commit: take the fusion cluster nearest the agent's final pose
        cl = S[s]["object_fusion"]["clusters"]; px, py = S[max(S)]["pose"][:2]
        if not cl: return None, "none", "", "no planner_commit and no fusion cluster at the decision step"
        c = min(cl, key=lambda c: np.hypot(c["centroid"][0] - px, c["centroid"][1] - py))
        commit = {"centroid": c["centroid"]}; far = 60
        why0 = f"no commit (suspicious-object search): cluster {c['id']} nearest the final pose ({np.hypot(c['centroid'][0] - px, c['centroid'][1] - py):.1f} m). "
    else: why0 = ""

    def get(t, k):   # mask of detection k at step t
        d = S[t]["detections"][k]
        if planner == "apexnav":
            p = f"{rec}masks/{d['mask']}"
            return read_mask(p, size) if os.path.exists(p) else None, ""
        comps = green_components(rec, t - 1)
        if not comps: return None, "no rgb_annot contour"
        return pick_component(comps, d, S[t]["detections"])
    why = why0
    mi, md = S[s]["planner_commit"].get("matched_detection_index") if S[s].get("planner_commit") else None, \
        (S[s]["planner_commit"] or {}).get("match_distance_m") if S[s].get("planner_commit") else None
    if mi is not None and md is not None and md <= MATCH_M:
        m, nt = get(s, mi)
        if m is not None: return m, planner + ("_det" if planner == "apexnav" else "_annot") + "_at_decision", 0, f"match {md} m. {nt}".strip()
        why = "matched detection has no mask/contour. " + nt
    elif mi is not None: why = f"matched detection {md} m from the cluster. "
    cands = []
    for t in S:
        if abs(t - s) > far: continue
        for k, d in enumerate(S[t]["detections"]):
            c = d.get("world_centroid")
            if c is None: continue
            dist = float(np.hypot(c[0] - commit["centroid"][0], c[1] - commit["centroid"][1]))
            if dist <= MATCH_M: cands.append((abs(t - s), dist, t, k))
    for _, dist, t, k in sorted(cands):
        m, nt = get(t, k)
        if m is not None:
            return m, planner + ("_det" if planner == "apexnav" else "_annot") + "_nearest_step", s - t, f"{why}cluster dist {dist:.2f} m at step {t}. {nt}".strip()
    return None, "none", "", why + f"no detection within {MATCH_M} m of the committed cluster within {far} steps carries a mask"


def gate_mask(planner, src, size):
    """gate arm: mask of the last accepted request + the image it was judged on (checked against the decision frame)."""
    d = src.split("/records/")[0] if "/records/" in src else src.split("/verification/")[0]
    a = [r for r in map(json.loads, open(d + "/verification/decisions.jsonl")) if r.get("state") == 1][-1]
    if a.get("approach") is None:
        vd = fs.accepted(d); img, mk = vd + "/input.jpg", vd + "/mask.png"
    else:
        ap = max(glob.glob(f"{d}/verification/approach/epi*_c{a['cluster']['id']}_s*"), key=lambda p: int(p.rsplit("_s", 1)[1]))
        v = a["approach"]["views"][-1]; img = glob.glob(f"{ap}/frames/f{v:03d}_a*.jpg")[0]; mk = img[:-4] + "_mask.png"
    if not os.path.exists(mk): return None, "none", "", "accepted view has no mask file"
    A_, B_ = Image.open(img).convert("RGB"), Image.open(src).convert("RGB")
    A_ = A_.resize(B_.size)   # later runs store the judged view at half resolution
    diff = np.abs(np.asarray(A_).astype(int) - np.asarray(B_).astype(int)).mean(-1)
    # VLFM gate input.jpg carries a green contour: compare outside the contour pixels
    diff = float(np.median(diff)) if planner == "vlfm" else float(diff.mean())
    if diff > ALIGN: return read_mask(mk, size), "gate_accept", "?", f"view != decision frame (diff {diff:.1f}): {os.path.relpath(img, d)}"
    return read_mask(mk, size), "gate_accept", 0, f"view diff {diff:.1f}"


def box_frame(o):
    """Image the index row's box/mask belongs to (box_xyxy is in decision-frame pixels): decision rgb minus frame_offset, or the
    judged gate view (path in the note) when the offset is '?'. Resize to the decision frame size before cropping."""
    off, src = o["frame_offset"], o["decision_frame"]
    if off == "0": return src
    if off == "?": return src.split("/records/")[0] + "/" + o["note"].rsplit(": ", 1)[1]
    d, f = os.path.split(src); return f"{d}/{int(f[:-4]) - int(off):06d}.jpg"


def build():
    A = {(r["raw_run"], int(r["episode"])): r for r in rows(ALL) if r["group"] != "posctrl"}
    os.makedirs(OUT + "png", exist_ok=True); out = []
    for r in rows(KEY):
        src = os.path.normpath(r["src"])
        if r["arm"] != "gate":   # the key's frame predates the ApexNav index fix: use the current decision frame when it is stored
            try: src = os.path.normpath(fs.decision_frame(r["run"], int(r["episode"]))[0])
            except FileNotFoundError: pass
        size = Image.open(src).size
        rec = src.split("/rgb/")[0] + "/" if "/rgb/" in src else None
        try:
            if r["arm"] == "gate": m, source, off, note = gate_mask(r["planner"], src, size)
            else: m, source, off, note = raw_mask(r["planner"], rec, int(os.path.basename(src)[:-4]), size)
        except Exception as e: m, source, off, note = None, "none", "", "error: " + repr(e)[:150]
        box = mask_box(m) if m is not None else None
        if m is not None and box is None: m, source, off, note = None, "none", "", note + " empty mask"
        a = A.get((r["run"], int(r["episode"])))
        if a and a["eye_box_norm"] and (m is None or off not in (0, "?")):   # by-eye box (labelled on eye_frame): beats a mask from a farther frame
            eo = int(os.path.basename(src)[:-4]) - int(a["eye_frame"][:6])
            if m is None or abs(eo) < abs(off):
                x0, y0, x1, y1 = map(float, a["eye_box_norm"].split())
                box = [round(x0 * size[0]), round(y0 * size[1]), round(x1 * size[0]), round(y1 * size[1])]
                note, m, source, off = (note + f" eye_box on {a['eye_frame']}" + ("" if m is None else f" replaces mask at offset {off}")).strip(), None, "eye_box", eo
        if m is None and source != "eye_box":   # box-only fallback: consensus box, only if on the decision frame itself
            a = A.get((r["run"], int(r["episode"])))
            if a and a["box_xyxy"] and os.path.normpath(ROOT + "/" + a["frame_path"]) == src:
                box, source, off, note = [int(v) for v in a["box_xyxy"].split()], "consensus_yolo", 0, note + f" box_source={a['box_source']}"
        path = ""
        if m is not None:
            path = f"{OUT}png/{r['id']}.png"; Image.fromarray(m.astype(np.uint8) * 255).save(path)
        out.append({"id": r["id"], "run": r["run"], "episode": r["episode"], "kind": r["kind"], "arm": r["arm"], "goal": r["goal"], "decision_frame": src,
                    "mask_path": path, "box_xyxy": " ".join(map(str, box)) if box else "", "source": source, "frame_offset": off, "note": note.strip()})
    with open(OUT + "index.tsv", "w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t"); w.writeheader(); w.writerows(out)
    cov = collections.Counter()
    for o in out:
        o_, s_ = o["frame_offset"], o["source"]
        kind_ = "mask" if o["mask_path"] else "box"
        c = "none" if s_ == "none" else f"exact {kind_}" if o_ == 0 else f"offset<=3 {kind_}" if o_ != "?" and abs(int(o_)) <= 3 else f"offset>3/unaligned {kind_}"
        cov[(o["run"], o["kind"], c)] += 1
    with open(OUT + "coverage.tsv", "w") as f:
        f.write("run\tkind\tcategory\tn\n"); [f.write("\t".join(map(str, (*k, v))) + "\n") for k, v in sorted(cov.items())]
    with open(OUT + "missing.tsv", "w") as f:
        f.write("run\tepisode\tkind\tnote\n"); [f.write(f"{o['run']}\t{o['episode']}\t{o['kind']}\t{o['note']}\n") for o in out if o["source"] == "none"]
    print(open(OUT + "coverage.tsv").read())


def sheets():
    import random
    I = rows(OUT + "index.tsv"); rnd = random.Random(0)
    sel = rnd.sample(I, 24); seen = {o["id"] for o in sel}
    sel += [o for o in I if o["frame_offset"] not in ("0", "") and o["id"] not in seen]
    per = 12; k = 0
    from PIL import ImageDraw
    for i in range(0, len(sel), per):
        tiles = []
        for o in sel[i:i + per]:
            im = np.asarray(Image.open(o["decision_frame"]).convert("RGB")).copy()
            if o["mask_path"]:
                cs, _ = cv2.findContours((np.asarray(Image.open(o["mask_path"])) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, (255, 0, 255), 3)
            elif o["box_xyxy"]:
                x0, y0, x1, y1 = map(int, o["box_xyxy"].split()); cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 3)
            t = Image.new("RGB", (im.shape[1], im.shape[0] + 34), "white"); t.paste(Image.fromarray(im), (0, 34))
            ImageDraw.Draw(t).text((3, 2), f"{o['id']} {o['run'][4:]} ep{o['episode']} {o['goal']} {o['kind']}\n{o['source']} off={o['frame_offset']}", fill="black")
            tiles.append(t.resize((t.width // 2, t.height // 2)))
        w, h = tiles[0].size; cols = 4; sheet = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), "white")
        for j, t in enumerate(tiles): sheet.paste(t, ((j % cols) * w, (j // cols) * h))
        sheet.save(f"{OUT}check_{k}.jpg", quality=90); k += 1
    print(len(sel), "items on", k, "sheets")


if __name__ == "__main__":
    {"index": build, "sheets": sheets}[sys.argv[1]]()
