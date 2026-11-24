"""Judge-free severity of false stops: a panel of 5 detectors independent of the planners/gate (no SAM3, GroundingDINO, YOLOv7, SigLIP, BLIP).

  sam3d-env sam3d python -m nav.eval.fp_consensus ITEMS.tsv OUT.tsv [--device cpu|cuda:N]

ITEMS.tsv: columns goal, frame_path, box_xyxy ("x1,y1,x2,y2" pixels); other columns are kept.
OUT.tsv = the rows + per detector: <name> = max GOAL-class score over detections with IoU >= 0.3 to the box (0 if none, nan if no box),
<name>_frame = max GOAL-class score anywhere in the frame (box-free).
COCO detectors give nan for a goal outside COCO. Thresholds are low (0.01) so scores are continuous."""
import argparse, csv, re, time
import numpy as np, torch
from PIL import Image

IOU, THR, W = 0.3, 0.01, "/data/cache/ultra/"   # ultralytics weights (yolov8x-worldv2.pt, rtdetr-l.pt)
COCO = {"chair": "chair", "couch": "couch", "sofa": "couch", "bed": "bed", "toilet": "toilet", "tv": "tv", "tv_monitor": "tv",
        "tv monitor": "tv", "plant": "potted plant", "potted plant": "potted plant", "potted_plant": "potted plant"}
NAMES = ["owlv2", "yoloworld", "detr", "rtdetr", "frcnn"]


def iou(a, B):
    B = np.asarray(B, float).reshape(-1, 4)
    w = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i)


def best(box, boxes, scores):
    """max score among detections with IoU >= IOU to box"""
    if len(scores) == 0: return 0.0
    s = np.asarray(scores, float)[iou(box, boxes) >= IOU]
    return float(s.max()) if len(s) else 0.0


def load(device):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection, DetrImageProcessor, DetrForObjectDetection
    from ultralytics import YOLOWorld, RTDETR
    import torchvision
    m = {}
    p, o = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"), Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()

    def owl(img, goal, coco):
        x = p(text=[[f"a photo of a {goal}"]], images=img, return_tensors="pt").to(device)
        with torch.no_grad(): r = o(**x)
        s = max(img.size)   # OWLv2 pads to a square at the bottom/right; boxes are cxcywh relative to it
        b = r.pred_boxes[0].cpu().numpy() * s
        return np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], r.logits[0, :, 0].sigmoid().cpu().numpy()
    m["owlv2"] = owl
    yw = YOLOWorld(W + "yolov8x-worldv2.pt")

    def world(img, goal, coco):
        yw.set_classes([goal])
        r = yw.predict(img, conf=THR, device=device, verbose=False)[0].boxes
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy()
    m["yoloworld"] = world
    dp, dm = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"), DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()
    l2i = {v: k for k, v in dm.config.id2label.items()}

    def detr(img, goal, coco):
        with torch.no_grad(): r = dm(**dp(images=img, return_tensors="pt").to(device))
        b = r.pred_boxes[0].cpu().numpy() * np.array(img.size * 2)   # cxcywh relative -> pixels
        return np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], r.logits[0].softmax(-1)[:, l2i[coco]].cpu().numpy()
    m["detr"] = detr
    rt = RTDETR(W + "rtdetr-l.pt")

    def rtdetr(img, goal, coco):
        r = rt.predict(img, conf=THR, device=device, verbose=False)[0].boxes
        k = [rt.names[int(c)] == coco for c in r.cls]
        return r.xyxy.cpu().numpy()[k], r.conf.cpu().numpy()[k]
    m["rtdetr"] = rtdetr
    wt = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    fr = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=wt, box_score_thresh=THR).to(device).eval()
    cats = wt.meta["categories"]

    def frcnn(img, goal, coco):
        with torch.no_grad(): r = fr([torchvision.transforms.functional.to_tensor(img).to(device)])[0]
        k = np.array([cats[i] == coco for i in r["labels"].cpu().tolist()], bool)
        return r["boxes"].cpu().numpy()[k], r["scores"].cpu().numpy()[k]
    m["frcnn"] = frcnn
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(8)
    rd = csv.DictReader(open(a.items), delimiter="\t"); items, fields = list(rd), rd.fieldnames + NAMES + [n + "_frame" for n in NAMES]
    models, cache, t = load(a.device), {}, {n: [] for n in NAMES}
    items = [r for r in items if r["frame_path"]]
    for r in items:
        box = np.array(re.findall(r"-?\d+\.?\d*", r["box_xyxy"]), float)
        goal = r["goal"].lower(); coco = COCO.get(goal)
        img = Image.open(r["frame_path"]).convert("RGB")
        for n in NAMES:
            if coco is None and n in ("detr", "rtdetr", "frcnn"): r[n] = r[n + "_frame"] = "nan"; continue
            k = (n, r["frame_path"], goal)
            if k not in cache:
                t0 = time.time(); cache[k] = models[n](img, goal.replace("_", " "), coco); t[n].append(time.time() - t0)
            r[n] = f"{best(box, *cache[k]):.4f}" if len(box) == 4 else "nan"
            r[n + "_frame"] = f"{max(cache[k][1], default=0.0):.4f}"   # box-free: max goal score anywhere in the frame
    w = csv.DictWriter(open(a.out, "w"), fields, delimiter="\t"); w.writeheader(); w.writerows(items)
    print({n: f"{np.mean(v):.2f}s/img" for n, v in t.items() if v})


if __name__ == "__main__": main()
