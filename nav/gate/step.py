"""ZSONStep: the planner-agnostic verification step (no ROS, no planner imports; see nav/README.md).

The planner keeps its exploration, its detector and its navigation. The step owns everything about *whether a candidate
detection is the target*: an object memory built from the planner's target-class masks, the verdicts, the evidence and
the approach that gets a clearer view first.

    step = ZSONStep(cfg, camera, log_dir)
    step.start_episode(run_index, scene, episode_id, label, goal_object_ids, max_episode_steps, sim)
    reopened = step.observe(step, rgb, depth, pose, masks, scores, camera_pitch)   # after every executed step
    verdict = step.check(step, point)           # the planner believes the target is at world point (x, y)
    ... verdict.kind: ACCEPT | REJECT | DEFER | GOTO | CONTROL
    out = step.act(step, rgb, depth, pose, masks)   # while CONTROL: an action name, or the final Verdict
    counts = step.finish_episode()

Objects: every target-class detection gets a world centroid (mask + depth + pose) and joins the nearest object within
``match_distance_m`` (else it opens one; an object's centroid is the mean of its detections). The views are kept in one
``EvidenceStore`` (per 0.75 m cell), each with a snapshot of the objects' centroids at its step; an object keeps its verdict
state, the views it used, its checks and its view bank. A check resolves the point to the nearest object within twice
``match_distance_m`` and judges the most readable view of it (preferring views that are new since its last check) with the
Verifier server (selector ``jev_sam3``); a view already judged reuses its verdict. correct -> ACCEPT, wrong -> REJECT,
insufficient -> an approach (GOTO: the planner's own navigation drives toward the object, CONTROL: ``act`` drives with
LiveApproach, nav.gate.approach, until it commits), budget permitting, else REJECT. A point without an object is DEFERRED
(a ``final`` check rejects it). A check on an object under approach is the planner reporting back (``reporting``): CONTROL
once the object is close and in view or the planner's navigation ``stalled``, else GOTO again; ``approach_ready`` says when
a new view of the approached object makes a report worthwhile.

Re-check: a rejected (or deferred) object is reset to unverified when ``recheck_cooldown_steps`` have passed since its last
check, it has fewer than ``max_checks`` checks, and a view of it is new (its camera ``recheck_min_distance_m`` from every
view used before, or seen from a direction ``recheck_min_angle_deg`` apart). ``observe`` returns those objects (id and world
point): a planner that keeps its own blacklist of rejected targets forgets them; one without ignores the return value.

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
from nav.gate.evidence import MATCH_DISTANCE_M, EvidenceStore, mask_touches_border, view_quality
from nav.gate.ground_truth import semantic_goal_overlap, slugify_label

DEFAULTS = {
    "url": None,                     # the Verifier server (perception.server --selector jev_sam3); required
    "timeout_s": 120.0,
    "ping_timeout_s": 60.0,          # long enough to wait out one in-flight verification
    "match_distance_m": MATCH_DISTANCE_M,   # detection -> object, and a check's point -> object (twice)
    "evidence_per_cell": 8,          # views kept per 0.75 m cell (the most readable plus the newest) ...
    "max_evidence": 256,             # ... and in total
    "view_bank": 24,                 # views of an object under approach kept for the local planner
    "sibling_m": 0.5,                # a nearer object this close to the approached one is a fragment of it (bank)
    "max_checks": 3,
    "recheck_cooldown_steps": 25,
    "recheck_min_distance_m": 1.0,
    "recheck_min_angle_deg": 45.0,
    "approach_max_per_object": 2,
    "approach_max_steps": 40,        # actions per approach
    "approach_episode_steps": 120,   # approach actions per episode
    "approach_min_closer_m": 0.5,    # a new view of an approached object: this much closer than every judged view ...
    "approach_min_angle_deg": 30.0,  # ... or this far around the object from each
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


def _bearing(camera, point):
    return math.atan2(point[1] - camera[1], point[0] - camera[0])


def _angle_between(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


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
        self.evidence = EvidenceStore(self.cfg["match_distance_m"], self.cfg["evidence_per_cell"],
                                      self.cfg["max_evidence"])
        self.verdicts = {}         # (step, index) of a judged view -> the server's verdict
        self.decided = []          # (points, verdict) of the objects an approach decided (sim frame): same-object identity
        self.ready = []            # objects under approach with a new view (approach_ready)
        self.control = self.live = None
        self.approach_steps_used = 0
        self.counts = {key: 0 for key in (
            "requests", "accepted", "rejected", "deferred", "no_evidence", "cached", "reopened", "same_object",
            "insufficient", "moves", "approach_early", "approach_arrived", "approach_failed", "approach_accepted",
            "approach_rejected", "approach_undecided", "approach_exhausted", "approach_steps", "sam3_decisions",
            "jev_decisions")}

    def finish_episode(self):
        return dict(self.counts)

    def pop_events(self):
        """Decisions and reopens since the last call, for the planner's step record."""
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
        metres (H x W), ``masks`` / ``scores`` / ``semantic_instances`` one entry per detection. Returns the objects the
        re-check rule reset to unverified: [{"object": id, "point": (x, y)}] (none while CONTROL drives)."""
        depth = np.asarray(depth, np.float32)
        centroids, new = [], []
        for index, raw in enumerate(masks):
            mask = (np.asarray(raw).squeeze() > 0).astype(np.uint8)
            area = int(mask.sum())
            centroid = self._centroid(depth, mask, pose) if area else None
            centroids.append(centroid)
            if centroid is None:
                continue
            score = float(scores[index])
            self._associate(centroid)
            new.append({"step": int(step), "index": index, "rgb": rgb, "mask": mask, "score": score,
                        "area_fraction": area / float(mask.size),
                        "quality": view_quality(score, area / float(mask.size), mask_touches_border(mask)),
                        "centroid": centroid, "camera": (float(pose[0]), float(pose[1])), "pitch": float(camera_pitch),
                        "semantic_instances": None if semantic_instances is None else semantic_instances[index]})
        snapshot = {o["id"]: o["centroid"] for o in self.objects}   # the objects as they stand at this step
        for view in new:
            view["objects"] = snapshot
            self.evidence.add(view)
        self._bank_views(new)
        self.last = {"depth": depth, "masks": list(masks), "centroids": centroids, "pose": pose}
        if self.control is not None:   # CONTROL drives: evidence only, no reopens and no approach views
            return []
        self._approach_views(step, new)
        return self._reopen(step, new, snapshot)

    def _associate(self, centroid):
        near = [o for o in self.objects if _distance(o["centroid"], centroid) <= self.cfg["match_distance_m"]]
        if not near:
            self.objects.append({"id": len(self.objects), "centroid": centroid, "n": 1, "state": None, "checks": 0,
                                 "last_step": 0, "point": centroid, "used": [], "approaches": 0, "pending": None,
                                 "bank": []})
            return
        obj = min(near, key=lambda o: _distance(o["centroid"], centroid))
        obj["centroid"] = tuple((a * obj["n"] + b) / (obj["n"] + 1) for a, b in zip(obj["centroid"], centroid))
        obj["n"] += 1

    def _object_at(self, point):
        """The nearest object within twice ``match_distance_m`` of ``point``, else None."""
        near = [o for o in self.objects if _distance(o["centroid"], point) <= 2 * self.cfg["match_distance_m"]]
        return min(near, key=lambda o: _distance(o["centroid"], point)) if near else None

    def _novel(self, view, used):
        """True when ``view`` sees its object from far enough, or a different enough direction, compared with every
        view used before."""
        bearing = _bearing(view["camera"], view["centroid"])
        return all(_distance(view["camera"], u["camera"]) >= self.cfg["recheck_min_distance_m"]
                   or math.degrees(_angle_between(bearing, u["bearing"])) >= self.cfg["recheck_min_angle_deg"]
                   for u in used)

    @staticmethod
    def _reference(view, obj, centroid, by_position=False):
        """Where ``view`` should be for ``obj``: the object's centroid at the view's step (objects grow), else
        ``centroid``. A stop (``by_position``) uses ``centroid``, the point."""
        return centroid if by_position else view["objects"].get(obj["id"], centroid)

    @staticmethod
    def _closer_to_other(view, obj, distance):
        """True when another object (as of the view's step) is nearer the view than ``distance``."""
        return any(other != obj["id"] and _distance(view["centroid"], c) < distance
                   for other, c in view["objects"].items())

    def _reopen(self, step, new, snapshot):
        reopened = []
        for obj in self.objects:
            if (obj["state"] not in (REJECT, DEFER) or obj["checks"] >= self.cfg["max_checks"]
                    or step - obj["last_step"] < self.cfg["recheck_cooldown_steps"]):
                continue
            centroid = snapshot.get(obj["id"], obj["point"])
            if any(_distance(v["centroid"], self._reference(v, obj, centroid)) <= self.cfg["match_distance_m"]
                   and self._novel(v, obj["used"]) for v in new):
                obj["state"] = None
                reopened.append({"object": obj["id"], "point": centroid})
                self.counts["reopened"] += 1
                self.events.append({"event": "reopen", "object": obj["id"], "step": int(step)})
                print(f"[verification] object {obj['id']} reopened at step {step} (new view)")
        return reopened

    def _select(self, obj, point, by_position):
        """(view, match distance) for ``obj``: the most readable view within ``match_distance_m`` of it (ties: nearest),
        else the nearest within twice that, else (None, None). Views nearer another object do not count. A re-check
        prefers views that are new by the re-check rule."""
        pool = list(self.evidence)
        if obj["used"]:
            pool = [v for v in pool if self._novel(v, obj["used"])] or pool

        def distance(view):
            return _distance(view["centroid"], self._reference(view, obj, point, by_position))

        if not by_position:
            pool = [v for v in pool if not self._closer_to_other(v, obj, distance(v))]
        matched = [v for v in pool if distance(v) <= self.cfg["match_distance_m"]]
        if matched:
            best = max(matched, key=lambda v: (v["quality"], -distance(v)))
            return best, distance(best)
        near = [v for v in pool if distance(v) <= 2 * self.cfg["match_distance_m"]]
        if near:
            best = min(near, key=distance)
            return best, distance(best)
        return None, None

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
        and the planner reports back: a new view of it (``approach_ready``), or ``stalled`` ("arrived", or "failed": no
        path, budget), so CONTROL takes over wherever it stands. ``meta``: the request, echoed in the log."""
        point = (float(point[0]), float(point[1]))
        if reporting:
            pending = [o for o in self.objects if o["pending"] is not None
                       and _distance(o["pending"]["object"], point) <= 2 * self.cfg["match_distance_m"]]
            if not pending:
                raise RuntimeError(f"check(reporting=True) at {point}: no approach is pending there")
            return self._continue_approach(step, pending[0], stalled)
        obj = self._object_at(point)
        record = {**{k: self.episode[k] for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "cluster": meta, "object": None if obj is None else obj["id"],
                  "point": list(point), "final": bool(final), "server": self.server.get("settings"),
                  "camera": self.camera, "evidence": None, "verdict": None, "latency_s": None, "error": None,
                  "policy": None, "source": "server", "ground_truth": None, "status": None, "model": None}
        self.counts["requests"] += 1
        if obj is not None:
            obj["checks"] += 1
            obj["last_step"], obj["point"] = int(step), point
            if obj["pending"] is not None:   # a fresh trigger: the planner dropped the approach
                print(f"[verification] object {obj['id']}: stale approach {obj['pending']['id']} dropped")
                obj["pending"] = None
        same = self._same_as_decided(point)
        if same is not None:
            self.counts["same_object"] += 1
            record.update(policy="same_as_" + ("accepted" if same[0] == ACCEPT else "rejected"),
                          same_iou=round(float(same[1]), 3))
            return self._decide(record, obj, same[0])
        view, match = (None, None) if obj is None else self._select(obj, point, final)
        if view is None:
            self.counts["no_evidence"] += 1
            record["policy"] = "on_no_evidence=defer"
            return self._decide(record, obj, REJECT if final else DEFER)
        return self._judge(record, obj, view, match, step, final)

    def _judge(self, record, obj, view, match, step, final):
        elevation = -math.degrees(view["pitch"])  # Habitat pitch is negative looking down
        record["evidence"] = {"step": view["step"], "index": view["index"], "score": view["score"],
                              "area_fraction": round(view["area_fraction"], 5), "match_distance_m": round(match, 3),
                              "camera": list(view["camera"]), "centroid": list(view["centroid"]),
                              "pitch": view["pitch"], "input_elevation": elevation}
        record["ground_truth"] = semantic_goal_overlap(view["semantic_instances"], self.episode["goal_object_ids"],
                                                       self.cfg["ground_truth_semantic_overlap"])
        obj["used"].append({"camera": view["camera"], "bearing": _bearing(view["camera"], view["centroid"]),
                            "range": _distance(view["camera"], view["centroid"])})
        key = (view["step"], view["index"])
        reply = None
        if key in self.verdicts:   # this exact view was already judged
            self.counts["cached"] += 1
            record.update(source="cached", verdict=self.verdicts[key])
        else:
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
            self.verdicts[key] = reply["verdict"]
            record.update(verdict=reply["verdict"], latency_s=round(time.time() - start, 3))
        verdict = record["verdict"]
        status = verdict.get("status")
        if status not in TRI_STATES:
            raise RuntimeError(f"verdict without a three-way status: {status!r}")
        model = verdict.get("source")
        record.update(status=status, model=model)
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
        pending = obj["pending"] = {"id": obj["approaches"], "issued_step": int(step), "object": tuple(view["centroid"]),
                                    "camera": tuple(view["camera"]), "max_steps": self._steps_allowed(step),
                                    "native": 0, "early": None, "meta": record["cluster"]}
        obj["bank"] = []   # seeded with every stored view of the object
        for item in sorted(self.evidence, key=lambda v: v["step"]):
            if self._bank_match(item, obj, pending) is not None:
                self._bank_add(obj, item)
        record["approach"] = {"id": pending["id"], "issued": True, "object": list(pending["object"]),
                              "camera": list(pending["camera"]), "max_steps": pending["max_steps"]}
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

    def _bank_match(self, view, obj, pending):
        """Match distance when ``view`` is a view of the approached object for the bank, else None: within
        ``match_distance_m`` of the issued object point or the object's centroid at the view's step, and no other
        object nearer it, unless that object lies within ``sibling_m`` of this one (a fragment of it)."""
        reference = view["objects"].get(obj["id"])
        distance = _distance(view["centroid"], pending["object"])
        if reference is not None:
            distance = min(distance, _distance(view["centroid"], reference))
        if distance > self.cfg["match_distance_m"]:
            return None
        reference = pending["object"] if reference is None else reference
        if any(other != obj["id"] and _distance(view["centroid"], c) < distance
               and _distance(c, reference) > self.cfg["sibling_m"] for other, c in view["objects"].items()):
            return None
        return distance

    def _bank_add(self, obj, view):
        """Add ``view`` to the object's bank once; past ``view_bank`` views drop the least readable one, never the newest."""
        bank = obj["bank"]
        if any((b["step"], b["index"]) == (view["step"], view["index"]) for b in bank):
            return
        bank.append(view)
        while len(bank) > int(self.cfg["view_bank"]):
            newest = max(bank, key=lambda b: (b["step"], b["index"]))
            bank.remove(min((b for b in bank if b is not newest), key=lambda b: (b["quality"], b["step"])))

    def _bank_views(self, new):
        """Bank this step's views of every object under approach."""
        for obj in self.objects:
            pending = obj["pending"]
            if pending is None:
                continue
            for view in new:
                if view["step"] >= pending["issued_step"] and self._bank_match(view, obj, pending) is not None:
                    self._bank_add(obj, view)

    def _approach_view(self, view, obj, pending):
        """Match distance when ``view`` is a new view of the approached object: taken since the approach was issued, of
        the object point (not nearer another object), and closer by ``approach_min_closer_m`` than, or
        ``approach_min_angle_deg`` around the object from, every view judged for the object. Else None."""
        if view["step"] < pending["issued_step"]:
            return None
        # The issued object point, or the object's centroid at the view's step (a large object's centroid drifts)
        distance = _distance(view["centroid"], pending["object"])
        if obj["id"] in view["objects"]:
            distance = min(distance, _distance(view["centroid"], view["objects"][obj["id"]]))
        if distance > self.cfg["match_distance_m"] or self._closer_to_other(view, obj, distance):
            return None
        rng = _distance(view["camera"], view["centroid"])
        bearing = _bearing(view["camera"], view["centroid"])
        closer, angle = float(self.cfg["approach_min_closer_m"]), math.radians(self.cfg["approach_min_angle_deg"])
        if all(rng <= u.get("range", _distance(u["camera"], pending["object"])) - closer
               or _angle_between(bearing, u["bearing"]) >= angle for u in obj["used"]):
            return distance
        return None

    def _approach_views(self, step, new):
        """Objects under approach with a new view this step (once until it is reported): ``approach_ready``."""
        for obj in self.objects:
            pending = obj["pending"]
            if pending is None or pending["early"] is not None:
                continue
            views = [(v, d) for v in new for d in [self._approach_view(v, obj, pending)] if d is not None]
            if views:
                pending["early"] = max(views, key=lambda v: (v[0]["quality"], -v[1]))
                self.ready.append(obj["id"])
                print(f"[verification] object {obj['id']}: new view at step {step} during approach {pending['id']}")

    def approach_ready(self):
        """Ids of the objects under approach (GOTO) with a new view since the last call: the planner should report back
        (``check(..., reporting=True)``)."""
        ready, self.ready = self.ready, []
        return ready

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

    def _approach_verdict(self, obj, kind=None):
        """GOTO while the planner's own navigation approaches, CONTROL when the local planner can take over."""
        pending = obj["pending"]
        kind = kind or (CONTROL if self._close(pending) else GOTO)
        if kind == CONTROL:
            self.control, self.ready = obj, []
        return Verdict(kind, obj["id"], pending["object"], pending["camera"], pending["max_steps"], pending["id"])

    def _continue_approach(self, step, obj, stalled):
        pending = obj["pending"]
        if not stalled and not self._close(pending):
            pending["early"] = None   # the next new view reports again
            return self._approach_verdict(obj, GOTO)
        pending["native"] = int(step) - pending["issued_step"]   # the planner's own approach actions
        self.approach_steps_used += pending["native"]
        self.counts["approach_steps"] += pending["native"]
        self.counts["approach_early" if not stalled else "approach_arrived" if stalled == "arrived"
                    else "approach_failed"] += 1
        print(f"[verification] object {obj['id']}: planner approach {pending['id']} "
              f"{stalled or 'close'} after {pending['native']} steps -> {PLANNER}")
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
                PLANNER, log_dir=os.path.join(self.log_dir, "approach", tag), tag=tag, bank=self.control["bank"])
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
        obj["pending"], obj["last_step"], self.control, self.live = None, int(step), None, None
        live.close()
        print(f"[verification] approach ({PLANNER}) object {obj['id']}: {result['status']} by {result['by']} "
              f"after {steps} actions")
        return self._decide(record, obj, kind)

    # ------------------------------------------------------------------
    # Decisions and logs
    # ------------------------------------------------------------------

    def _decide(self, record, obj, kind, view=None, reply=None, verdict=None):
        if obj is not None:
            obj["state"] = kind
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
