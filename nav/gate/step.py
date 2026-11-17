"""ZSONStep: the planner-agnostic verification step (no ROS, no planner imports; see nav/README.md).

The planner keeps its exploration, its detector and its navigation. The step owns everything about *whether a candidate
detection is the target*: an object memory built from the planner's target-class masks, the verdicts, the evidence and
the approach that gets a clearer view first.

    step = ZSONStep(cfg, camera, log_dir)
    step.start_episode(run_index, scene, episode_id, label, goal_object_ids, max_episode_steps, sim)
    step.observe(step, rgb, depth, pose, masks, scores, camera_pitch)      # after every executed step
    verdict = step.check(step, point)           # the planner believes the target is at world point (x, y)
    ... verdict.kind: ACCEPT | REJECT | DEFER | GOTO | CONTROL
    out = step.act(step, rgb, depth, pose, masks)   # while CONTROL: an action name, or the final Verdict
    counts = step.finish_episode()

Objects: every target-class detection gets a world centroid (mask + depth + pose) and joins the nearest object within
``match_distance_m`` (else it opens one); an object keeps its most readable views (``view_bank``) and its verdict. A check
resolves the point to the nearest object within twice that distance. An object that already has a verdict answers from
memory; an undecided one is judged on its most readable view by the Verifier server (selector ``jev_sam3``): correct ->
ACCEPT, wrong -> REJECT, insufficient -> an approach (GOTO: the planner's own navigation drives toward the object, CONTROL:
``act`` drives with LiveApproach, nav.gate.approach, until it commits), budget permitting, else REJECT. A point without an
object is DEFERRED (a ``final`` check rejects it). A check on an object under approach is the planner reporting back:
CONTROL once the object is close and in view (``approach_ready``) or the planner's navigation ``stalled``, else GOTO again.

Server errors raise. Logs: ``<log_dir>/decisions.jsonl`` (one line per decision; ``meta`` of the check is echoed as
``cluster``) and, with ``save_images``, ``<log_dir>/epi<run>_<label>/obj<id>_s<step>/{input.jpg, mask.png, view.png,
decision.json}``.
"""
import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from nav.gate import client
from nav.gate.evidence import MATCH_DISTANCE_M, mask_touches_border, view_quality
from nav.gate.ground_truth import semantic_goal_overlap, slugify_label

DEFAULTS = {
    "url": None,                     # the Verifier server (perception.server --selector jev_sam3); required
    "timeout_s": 120.0,
    "ping_timeout_s": 60.0,          # long enough to wait out one in-flight verification
    "match_distance_m": MATCH_DISTANCE_M,   # detection -> object, and a check's point -> object (twice)
    "view_bank": 24,                 # views kept per object (the most readable plus the newest)
    "max_views": 256,                # views kept in total
    "approach_max_per_object": 2,
    "approach_max_steps": 40,        # actions per approach
    "approach_episode_steps": 120,   # approach actions per episode
    "save_images": True,
    "ground_truth_semantic_overlap": 0.5,
}
ACCEPT, REJECT, DEFER, GOTO, CONTROL = "ACCEPT", "REJECT", "DEFER", "GOTO", "CONTROL"
STATES = {ACCEPT: 1, REJECT: -1, DEFER: 2, GOTO: 3, CONTROL: 3}   # as logged (the planner's verification states)
COUNT = {ACCEPT: "accepted", REJECT: "rejected", DEFER: "deferred"}
TRI_STATES = ("correct", "wrong", "insufficient")
PLANNER = "mppi_v15"
MIN_PLANNER_STEPS = 12   # CONTROL gets at least this many actions after the planner's own approach
# CONTROL hand-off (_close): the object is within HANDOFF_M of the agent and detected (a target mask of >= HANDOFF_PX whose
# centroid is within HANDOFF_MATCH_M of the object point), or its evidence view was taken within HANDOFF_SEEN_M of the agent
HANDOFF_M, HANDOFF_MATCH_M, HANDOFF_PX, HANDOFF_SEEN_M = 3.0, 1.0, 200, 1.0


@dataclass(frozen=True)
class Verdict:
    """``kind``; ``object`` = the step's object id. GOTO / CONTROL carry the approach: ``point`` (the object), ``camera``
    (where the insufficient view was taken), ``max_steps``, ``approach_id`` (1-based per object)."""
    kind: str
    object: int = None
    point: tuple = None
    camera: tuple = None
    max_steps: int = None
    approach_id: int = None


def target_name(label):
    """The Verifier category for a planner label ("potted_plant", "table | desk")."""
    return str(label).split("|")[0].strip().replace("_", " ")


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ZSONStep:
    """``camera``: {hfov, width, height} of the planner's depth / RGB sensor (logged with every decision)."""

    def __init__(self, cfg, camera, log_dir, verify=None, ping=None):
        unknown = set(cfg or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown verification settings: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg or {}).items() if v is not None}}
        if not self.cfg["url"]:
            raise ValueError("verification.url is required")
        self.camera, self.log_dir = camera, log_dir
        self.fx = camera["width"] / (2 * math.tan(camera["hfov"] * np.pi / 360.0))
        self._verify = verify or client.verify
        self.server = (ping or client.ping)(self.cfg["url"], timeout=float(self.cfg["ping_timeout_s"]))
        selector = (self.server.get("settings") or {}).get("selector")
        if selector != "jev_sam3":
            raise ValueError(f"the verification step needs the server's selector jev_sam3, not {selector!r}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"[verification] ZSON step ON: {self.cfg['url']} {json.dumps(self.server.get('settings'))}")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, run_index, scene, episode_id, label, goal_object_ids, max_episode_steps, sim):
        """``sim``: the agent body the local approach planner reads (nav.approach.scene; habitat_sim.Simulator)."""
        self.max_episode_steps, self.sim = int(max_episode_steps), sim
        self.episode = {"run_index": int(run_index), "scene": str(scene), "episode_id": str(episode_id),
                        "label": str(label), "target": target_name(label),
                        "goal_object_ids": [str(i) for i in goal_object_ids if i is not None]}
        self.objects, self.events, self.last = [], [], None
        self.decided = []          # (points, verdict) of the objects an approach decided (sim frame): same-object identity
        self.control = self.live = None
        self.approach_steps_used = 0
        self.counts = {key: 0 for key in (
            "requests", "accepted", "rejected", "deferred", "no_evidence", "same_object", "insufficient", "moves",
            "approach_early", "approach_stalled", "approach_accepted", "approach_rejected", "approach_undecided",
            "approach_exhausted", "approach_steps", "sam3_decisions", "jev_decisions")}

    def finish_episode(self):
        return dict(self.counts)

    def pop_events(self):
        """Decisions since the last call, for the planner's step record."""
        events, self.events = self.events, []
        return events or None

    def accepted_points(self):
        """Points (sim frame) of every object an approach accepted, in decision order (the split navigator's goals)."""
        return [points for points, verdict in self.decided if verdict == ACCEPT]

    # ------------------------------------------------------------------
    # Object memory
    # ------------------------------------------------------------------

    def _centroid(self, depth, mask, pose):
        """World (x, y) of a detection: the mean of its (eroded) mask's depth points, as the planner computes it."""
        v, u = np.where(cv2.erode(mask.astype(np.uint8) * 255, None))
        if not len(v):
            return None
        z = depth[v, u]
        lateral = (u - depth.shape[1] // 2) * z / self.fx
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return (float(np.mean(c * z + s * lateral)) + float(pose[0]), float(np.mean(s * z - c * lateral)) + float(pose[1]))

    def observe(self, step, rgb, depth, pose, masks, scores, camera_pitch, semantic_instances=None):
        """Store this step's target-class detections. ``pose`` (x, y, yaw) in the planner's world frame, ``depth`` in
        metres (H x W), ``masks`` / ``scores`` / ``semantic_instances`` one entry per detection."""
        depth = np.asarray(depth, np.float32)
        centroids = []
        for index, raw in enumerate(masks):
            mask = (np.asarray(raw).squeeze() > 0).astype(np.uint8)
            area = int(mask.sum())
            centroid = self._centroid(depth, mask, pose) if area else None
            centroids.append(centroid)
            if centroid is None:
                continue
            score = float(scores[index])
            self._add({"step": int(step), "index": index, "rgb": rgb, "mask": mask, "score": score,
                       "area_fraction": area / float(mask.size),
                       "quality": view_quality(score, area / float(mask.size), mask_touches_border(mask)),
                       "centroid": centroid, "camera": (float(pose[0]), float(pose[1])), "pitch": float(camera_pitch),
                       "semantic_instances": None if semantic_instances is None else semantic_instances[index]})
        self.last = {"depth": depth, "masks": list(masks), "centroids": centroids, "pose": pose}

    def _add(self, view):
        near = [o for o in self.objects if _distance(o["centroid"], view["centroid"]) <= self.cfg["match_distance_m"]]
        if near:
            obj = min(near, key=lambda o: _distance(o["centroid"], view["centroid"]))
            n = obj["n"]
            obj["centroid"] = tuple((a * n + b) / (n + 1) for a, b in zip(obj["centroid"], view["centroid"]))
            obj["n"] = n + 1
        else:
            obj = {"id": len(self.objects), "centroid": view["centroid"], "n": 1, "views": [], "verdict": None,
                   "approaches": 0, "pending": None}
            self.objects.append(obj)
        obj["views"].append(view)
        if len(obj["views"]) > int(self.cfg["view_bank"]):
            self._drop_worst(obj["views"])
        while sum(len(o["views"]) for o in self.objects) > int(self.cfg["max_views"]):
            crowded = max(self.objects, key=lambda o: len(o["views"]))
            if len(crowded["views"]) < 2:
                break
            self._drop_worst(crowded["views"])

    @staticmethod
    def _drop_worst(views):
        """Drop the least readable view, never the newest."""
        newest = max(views, key=lambda v: (v["step"], v["index"]))
        views.remove(min((v for v in views if v is not newest), key=lambda v: (v["quality"], v["step"])))

    def _object_at(self, point):
        """The nearest object within twice ``match_distance_m`` of ``point``, else None."""
        near = [o for o in self.objects if _distance(o["centroid"], point) <= 2 * self.cfg["match_distance_m"]]
        return min(near, key=lambda o: _distance(o["centroid"], point)) if near else None

    def _same_as_decided(self, point):
        """The identity rule of the approach (approach.associate) against the objects an approach decided: each one's
        points projected into the last frame with the depth test (local_nav.track_mask) are its outline T; a target
        mask near ``point`` that matches T is that object. -> (verdict, iou) or None."""
        from nav.approach import scene
        from nav.approach.local_nav import track_mask
        from nav.gate.approach import associate
        if not self.decided or self.last is None:
            return None
        masks = [np.asarray(m).squeeze() > 0 for m, c in zip(self.last["masks"], self.last["centroids"])
                 if c is not None and _distance(c, point) <= HANDOFF_MATCH_M]
        if not masks:
            return None
        cam = scene.camera(self.sim)
        for points, verdict in reversed(self.decided):
            outline = track_mask(points, self.last["depth"], *cam)
            if not outline.any():
                continue
            best, scores = associate(masks, outline)
            if best is not None:
                return verdict, scores[best][1]
        return None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def check(self, step, point, final=False, reporting=False, stalled=False, meta=None):
        """The planner believes the target is at ``point`` (world x, y). ``final``: it is about to stop here (no approach
        possible: insufficient rejects, a deferral rejects). ``reporting``: ``point`` is the GOTO point the step gave it,
        and the planner reports back (a new view of it, or ``stalled``: its own navigation ended, arrived / no path /
        budget, so CONTROL takes over wherever it stands). ``meta``: the request, echoed in the log."""
        point = (float(point[0]), float(point[1]))
        if reporting:
            pending = [o for o in self.objects if o["pending"] is not None
                       and _distance(o["pending"]["object"], point) <= 2 * self.cfg["match_distance_m"]]
            if not pending:
                raise RuntimeError(f"check(reporting=True) at {point}: no approach is pending there")
            return self._continue_approach(step, pending[0], stalled)
        obj = self._object_at(point)
        if obj is not None and obj["pending"] is not None:   # a fresh trigger: the planner dropped the approach
            print(f"[verification] object {obj['id']}: stale approach {obj['pending']['id']} dropped")
            obj["pending"] = None
        record = {**{k: self.episode[k] for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "cluster": meta, "object": None if obj is None else obj["id"],
                  "point": list(point), "final": bool(final), "server": self.server.get("settings"),
                  "camera": self.camera, "evidence": None, "verdict": None, "latency_s": None, "error": None,
                  "policy": None, "source": "server", "ground_truth": None, "status": None, "model": None}
        self.counts["requests"] += 1
        if obj is not None and obj["verdict"] is not None:
            record.update(policy="remembered", source="memory")
            return self._decide(record, obj, obj["verdict"])
        same = self._same_as_decided(point)
        if same is not None:
            self.counts["same_object"] += 1
            record.update(policy="same_as_" + ("accepted" if same[0] == ACCEPT else "rejected"),
                          same_iou=round(float(same[1]), 3))
            return self._decide(record, obj, same[0])
        if obj is None:
            self.counts["no_evidence"] += 1
            record["policy"] = "on_no_evidence=defer"
            return self._decide(record, obj, REJECT if final else DEFER)
        return self._judge(record, obj, step, final)

    def _judge(self, record, obj, step, final):
        view = max(obj["views"], key=lambda v: (v["quality"], -_distance(v["centroid"], obj["centroid"])))
        elevation = -math.degrees(view["pitch"])  # Habitat pitch is negative looking down
        record["evidence"] = {"step": view["step"], "index": view["index"], "score": view["score"],
                              "area_fraction": round(view["area_fraction"], 5),
                              "match_distance_m": round(_distance(view["centroid"], record["point"]), 3),
                              "camera": list(view["camera"]), "centroid": list(view["centroid"]),
                              "pitch": view["pitch"], "input_elevation": elevation}
        record["ground_truth"] = semantic_goal_overlap(view["semantic_instances"], self.episode["goal_object_ids"],
                                                       self.cfg["ground_truth_semantic_overlap"])
        start = time.time()
        try:
            reply = self._verify(self.cfg["url"], view["rgb"], view["mask"], self.episode["target"],
                                 input_elevation=elevation,
                                 request_id=f"epi{self.episode['run_index']}_o{obj['id']}_s{step}",
                                 timeout=float(self.cfg["timeout_s"]))
        except Exception as error:   # no fallbacks: a run never goes on silently unverified
            record.update(error=f"{type(error).__name__}: {error}", latency_s=round(time.time() - start, 3))
            self._log(record, view, None)
            raise
        verdict = reply["verdict"]
        status = verdict.get("status")
        if status not in TRI_STATES:
            raise RuntimeError(f"verdict without a three-way status: {status!r}")
        model = verdict.get("source")
        record.update(verdict=verdict, latency_s=round(time.time() - start, 3), status=status, model=model)
        if model in ("sam3", "jev"):
            self.counts[f"{model}_decisions"] += 1
        if status != "insufficient":
            return self._decide(record, obj, ACCEPT if status == "correct" else REJECT, view, reply)
        self.counts["insufficient"] += 1
        blocked = self._blocked(obj, step, final)
        if blocked is not None:
            self.counts["approach_exhausted"] += 1
            record["policy"] = f"insufficient,{blocked}"
            return self._decide(record, obj, REJECT, view, reply)
        obj["approaches"] += 1
        self.counts["moves"] += 1
        obj["pending"] = {"id": obj["approaches"], "issued_step": int(step), "object": tuple(view["centroid"]),
                          "camera": tuple(view["camera"]), "max_steps": self._steps_allowed(step), "native": 0,
                          "meta": record["cluster"]}
        record["approach"] = {"id": obj["pending"]["id"], "issued": True, "object": list(obj["pending"]["object"]),
                              "camera": list(obj["pending"]["camera"]), "max_steps": obj["pending"]["max_steps"]}
        verdict = self._approach_verdict(obj)
        return self._decide(record, obj, verdict.kind, view, reply, verdict)

    def _remaining(self):
        return int(self.cfg["approach_episode_steps"]) - self.approach_steps_used

    def _steps_allowed(self, step):
        """Actions the next approach may take: per approach, per episode, and the steps left before the episode's
        step limit (the last step is the forced stop)."""
        return min(int(self.cfg["approach_max_steps"]), self._remaining(), self.max_episode_steps - int(step) - 1)

    def _blocked(self, obj, step, final):
        """Why an insufficient view cannot be approached (None: it can)."""
        if final:
            return "final"
        if obj["approaches"] >= int(self.cfg["approach_max_per_object"]):
            return "object_budget"
        if min(int(self.cfg["approach_max_steps"]), self._remaining()) <= 0:
            return "episode_budget"
        if self._steps_allowed(step) <= 0:
            return "episode_end"
        return None

    # ------------------------------------------------------------------
    # Approach
    # ------------------------------------------------------------------

    def _close(self, pending):
        """True when CONTROL can take ``pending`` from the last observed frame (see HANDOFF_*). Farther or out of view,
        the planner's own navigation approaches first: the local planner only knows what it has seen since it took over."""
        if self.last is None:
            return False
        obj, here = np.asarray(pending["object"], float), np.asarray(self.last["pose"][:2], float)
        if np.hypot(*(obj - here)) > HANDOFF_M:
            return False
        if np.hypot(*(np.asarray(pending["camera"], float) - here)) <= HANDOFF_SEEN_M:
            return True   # seen from where the agent stands (e.g. during a spin): the planner turns to it
        return any(c is not None and _distance(c, obj) <= HANDOFF_MATCH_M
                   and int((np.asarray(m).squeeze() > 0).sum()) >= HANDOFF_PX
                   for m, c in zip(self.last["masks"], self.last["centroids"]))

    def approach_ready(self):
        """True when an object under approach (GOTO) is now close and in view: the planner should check again."""
        return any(o["pending"] is not None and self._close(o["pending"]) for o in self.objects)

    def _approach_verdict(self, obj, kind=None):
        """GOTO while the planner's own navigation approaches, CONTROL when the local planner can take over."""
        pending = obj["pending"]
        kind = kind or (CONTROL if self._close(pending) else GOTO)
        if kind == CONTROL:
            self.control = obj
        return Verdict(kind, obj["id"], pending["object"], pending["camera"], pending["max_steps"], pending["id"])

    def _continue_approach(self, step, obj, stalled):
        pending = obj["pending"]
        if not stalled and not self._close(pending):
            return self._approach_verdict(obj, GOTO)
        pending["native"] = int(step) - pending["issued_step"]   # the planner's own approach actions
        self.approach_steps_used += pending["native"]
        self.counts["approach_steps"] += pending["native"]
        self.counts["approach_stalled" if stalled else "approach_early"] += 1
        print(f"[verification] object {obj['id']}: planner approach {pending['id']} "
              f"{'stalled' if stalled else 'close'} after {pending['native']} steps -> {PLANNER}")
        return self._approach_verdict(obj, CONTROL)

    def act(self, step, rgb, depth, pose, masks):
        """While CONTROL (the camera level, pitch 0): the frame of the step just executed -> the next action
        (``move_forward`` / ``turn_left`` / ``turn_right``), or the final Verdict (ACCEPT, REJECT, DEFER) once the
        approach commits."""
        from nav.gate.approach import LiveApproach
        if self.control is None:
            raise RuntimeError("act() without a CONTROL verdict")
        if self.live is None:
            pending = self.control["pending"]
            tag = f"epi{self.episode['run_index']}_o{self.control['id']}_s{step}"
            self.live_start = time.time()
            self.live = LiveApproach(
                self.sim, self.cfg["url"], self.episode["target"], pending["object"],
                min(max(pending["max_steps"] - pending["native"], MIN_PLANNER_STEPS),
                    self.max_episode_steps - 1 - int(step)),
                PLANNER, log_dir=os.path.join(self.log_dir, "approach", tag), tag=tag, bank=self.control["views"])
        depth = np.asarray(depth, np.float32)
        centroids = [self._centroid(depth, (np.asarray(m).squeeze() > 0).astype(np.uint8), pose) for m in masks]
        name = self.live.step(rgb, depth, masks, centroids, pose)
        return name if name is not None else self._finish(step)

    def end_approach(self, step):
        """The episode ended during CONTROL: commit what the approach has and return its Verdict."""
        self.live._forced("episode_over")
        return self._finish(step)

    def _finish(self, step):
        live, obj = self.live, self.control
        result = dict(live.result, seconds=round(time.time() - self.live_start, 1))
        pending = obj["pending"]
        steps = int(result["actions"])
        self.approach_steps_used += steps
        self.counts["approach_steps"] += steps
        # an approach that ended stuck brought no evidence it could act on: DEFER, never REJECT
        kind = ACCEPT if result["accept"] else DEFER if result["status"] == "undecided" or result.get("end") == "stuck" \
            else REJECT
        self.counts[{ACCEPT: "approach_accepted", REJECT: "approach_rejected", DEFER: "approach_undecided"}[kind]] += 1
        self.counts["sam3_decisions"] += result.get("by") in ("belief", "forced")
        points = live.object_points()
        if kind != DEFER and points is not None and len(points) >= 3:
            self.decided.append((np.asarray(points, np.float32), kind))   # a later check of the same object inherits it
        record = {**{k: self.episode[k] for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "cluster": pending["meta"], "object": obj["id"], "point": list(pending["object"]),
                  "final": False, "server": self.server.get("settings"), "camera": self.camera, "evidence": None,
                  "verdict": {"status": result["status"], "label": None}, "latency_s": result["seconds"],
                  "error": None, "source": "server", "ground_truth": None, "status": result["status"],
                  "model": "sam3", "policy": f"approach_planner,{result['by']},{kind.lower()}",
                  "approach": {"id": pending["id"], "planner": PLANNER, "status": result.get("end") or result["by"],
                               "steps": steps, "views": result.get("views"), "counted": result.get("counted"),
                               "object": list(pending["object"]), "issued_step": pending["issued_step"],
                               "log": live.log_dir, "episode_steps_used": self.approach_steps_used}}
        obj["pending"], self.control, self.live = None, None, None
        live.close()
        print(f"[verification] approach ({PLANNER}) object {obj['id']}: {result['status']} by {result['by']} "
              f"after {steps} actions")
        return self._decide(record, obj, kind)

    # ------------------------------------------------------------------
    # Decisions and logs
    # ------------------------------------------------------------------

    def _decide(self, record, obj, kind, view=None, reply=None, verdict=None):
        if obj is not None and kind in (ACCEPT, REJECT):
            obj["verdict"] = kind
        record["state"] = STATES[kind]
        if kind in COUNT:   # an approach is counted as a move when issued
            self.counts[COUNT[kind]] += 1
        self._log(record, view, reply)
        label = (record["verdict"] or {}).get("label")
        self.events.append({"event": "decision", "object": record["object"], "state": STATES[kind], "label": label,
                            "source": record["source"], "error": record["error"]})
        print(f"[verification] object {record['object']} {kind}: label {label!r} for {self.episode['target']!r} "
              f"({record['policy'] or record['source']}, {record['latency_s'] or 0.0:.1f} s)")
        return verdict or Verdict(kind, record["object"])

    def _log(self, record, view, reply):
        with open(os.path.join(self.log_dir, "decisions.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if not self.cfg["save_images"] or view is None:
            return
        folder = os.path.join(self.log_dir, f"epi{record['run_index']}_{slugify_label(record['label'])}",
                              f"obj{record['object']}_s{record['step']}")
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "input.jpg"), cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(folder, "mask.png"), view["mask"] * 255)
        if reply and reply.get("view_image"):
            cv2.imwrite(os.path.join(folder, "view.png"),
                        cv2.cvtColor(client.decode_png(reply["view_image"]), cv2.COLOR_RGB2BGR))
        with open(os.path.join(folder, "decision.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=1)
