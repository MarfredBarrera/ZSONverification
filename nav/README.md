# nav: the ZSON verification step, planner-agnostic

An ObjectNav planner keeps its exploration, its detector and its navigation. The **ZSON verification step**
(`nav/gate/step.py`, `ZSONStep`) decides **whether a candidate detection is the target**: it is fully isolated from the
planner's detector. It gets RGB, depth, pose and the camera intrinsics every step, plus one boolean (the planner's detector sees the target in
this frame), runs SAM3 itself on those frames (Verifier server `/detect`), builds its own object memory from that, keeps the verdicts and evidence on those objects, and when the first
view is insufficient it gets a clearer one (the approach). No ROS, no planner imports, no dependence on the planner's
detections or object clusters. ApexNav is the first adapter (`planners/ApexNav`, `habitat_evaluation.py`, a thin one); VLFM is next.

```
nav/gate      step.py (ZSONStep, Verdict), approach.py (LiveApproach: closed-loop approach + decision flow), client.py (HTTP to
              perception.server), evidence.py (view quality), ground_truth.py (logging helpers)
nav/approach  mppi.py (MppiV15 + closer.py warm start), decide.py (belief + fused SAM3 verdict), size_opinion.py,
              local_nav.py, geom.py, scene.py
nav/lastmile  optional, separate: split.py (SplitNav), medial.py, collision_map.py, body_map.py
nav/eval      closer_attrib.py, nav_table.py (outcome scoring of run directories)
```

## What is the step, what is the planner

| the step (ours) | the planner (native) |
|---|---|
| its own SAM3 detection of the flagged frames, and of every frame while CONTROL drives (`/detect`: the target plus the other goal categories chair, couch, bed, toilet, tv, potted plant; SAM3 score >= 0.3, the approach's threshold) | exploration, frontier / value maps |
| object memory: world centroid per target instance, association by distance (`match_distance_m` 0.75 m), evidence views per place, verdict state / used views / view bank per object; instances of the other categories are non-target objects (positions only: a view nearer another object does not count) | the detector (it flags frames with a target detection, and decides when to ask for a check) |
| verdicts and evidence (Verifier server, `jev_sam3`), identity of an already decided object (`same_as_decided`), budgets | |
| the approach: JEV naming belief, fused SAM3, MPPI view planner (`LiveApproach`, `MppiV15`) | choosing the target, its own navigation, STOP |

The step ends at the verdict plus the approach needed to reach it. **Split last-mile navigation** (`nav/lastmile`) is not
part of it: an adapter may call `SplitNav.act` with `ZSONStep.accepted_points()`; in the ApexNav adapter it is on by default
and `NAV_SPLIT=false` turns it off (ablation).

Requirements: numpy, scipy, cv2, torch (CPU) and the Verifier server (`python -m perception.server --selector jev_sam3`;
`perception.jev` and `perception.select` are imported by `nav.approach.decide`). Put the repo root on `PYTHONPATH`. The
local approach planner reads the agent body from a `habitat_sim.Simulator` (`start_episode(..., sim)`; `nav.approach.scene`
reads pose and camera from it, numpy-quaternion needed), with ApexNav's embodiment (640x480, hfov 79, camera 0.88 m, radius
0.18, forward 0.25 m, turns 30 deg, no sliding, camera pitch 0 during an approach). Another simulator or robot needs its
own `scene.agent_pose` / `scene.camera`; that is the one non-agnostic dependency left.

## Interface (`ZSONStep`)

Frames are in the **planner's** world frame: `pose = (x, y, yaw)`, `point = (x, y)`, depth in metres (`H x W`), RGB
`H x W x 3 uint8`. The planner gives the step no masks. Its one input besides RGB-D, pose and the trigger point is
`target_seen`: its own detector reports a target-class detection in this frame (VLFM: its detector's target detection).

| Call | When | Inputs | Output |
|---|---|---|---|
| `ZSONStep(cfg, camera, log_dir)` | once | `cfg` = `step.DEFAULTS` keys (`url`, `timeout_s`, `match_distance_m`, `view_bank`, `approach_max_per_object`, `approach_max_steps`, `approach_episode_steps`, `save_images`, ...); `camera` = `{hfov, width, height}` | pings the server; raises on a wrong URL or selector |
| `start_episode(run_index, scene, episode_id, label, goal_object_ids, max_episode_steps, sim)` | episode start | `label` the target category | |
| `observe(step, rgb, depth, pose, camera_pitch, target_seen, semantic=None)` | after every executed step (also during CONTROL) | the frame; `target_seen` (bool); `semantic` (scene instance ids) is for ground-truth logging only; the step runs SAM3 on the RGB (one `/detect` call) when `target_seen` or while CONTROL drives | the objects the re-check rule reset to unverified: `[{object, point}]` (a planner with its own blacklist forgets them there; others ignore it) |
| `check(step, point, final=False, reporting=False, stalled=False, meta=None)` | the planner believes the target is at `point` | `final`: it is about to stop here (no approach possible); `reporting`: `point` is the GOTO point the step gave it and the planner reports back; `stalled`: `"arrived"` or `"failed"`, its own navigation ended; `meta`: echoed in the log | `Verdict` |
| `approach_ready()` | after `observe`, while a GOTO is out | | ids of the GOTO objects with a new view (0.5 m closer or 30 deg around from every judged view, once each): the planner should report back (`reporting=True`); CONTROL takes over then if the object is close and in view, else GOTO goes on |
| `act(step, rgb, depth, pose)` | every step while CONTROL (camera level) | the frame of the step just executed (already `observe`d: its SAM3 masks are used) | an action name (`move_forward` / `turn_left` / `turn_right`) or the final `Verdict` (ACCEPT, REJECT, DEFER) |
| `end_approach(step)` | the episode ended during CONTROL | | the final `Verdict` |
| `accepted_points()` | for `SplitNav` | | points of the objects an approach accepted |
| `finish_episode()`, `pop_events()` | episode end / per step | | counts, decision events for the step record |

`Verdict.kind`:

| kind | meaning for the planner |
|---|---|
| `ACCEPT` | the target is verified: stop / go on to it |
| `REJECT` | not the target (the planner drops it; the re-check rule may reopen it) |
| `DEFER` | no evidence yet (no object at the point): treat as a non-detection and keep exploring; a `final` check rejects |
| `GOTO` | insufficient view: navigate toward `verdict.point` with the planner's own navigation (`camera` = where the unclear view was taken, `max_steps`); when `approach_ready()` or the navigation ends, call `check(..., reporting=True[, stalled=True])` |
| `CONTROL` | the step's local planner takes over: level the camera, then `act` every step until it returns a Verdict |

A check judges the most readable view of the object (views new since its last check preferred; a view already judged reuses its
verdict, no server call): correct -> ACCEPT, wrong -> REJECT, insufficient -> GOTO if the object is far or out of view, else
CONTROL; budgets: `approach_max_per_object`, `approach_max_steps`, `approach_episode_steps`, the episode limit; no budget, or
`final` -> REJECT. An approach that finds no evidence (or gets stuck) ends in DEFER. **Re-check**: a rejected or deferred object
is reset to unverified when `recheck_cooldown_steps` (25) have passed since its last check, it has fewer than `max_checks` (3)
checks, and a new view of it comes from a camera `recheck_min_distance_m` (1.0) from every view used before or from a direction
`recheck_min_angle_deg` (45) apart; `observe` returns it. Server errors raise.

The decision flow in CONTROL: JEV judges only the uncertainty of the first frame; a sure frame is decided by SAM3's raw label;
an unsure one makes `LiveApproach` (`MppiV15` on the agent's own depth map) collect views (`decide.py`: JEV naming belief, new
viewpoints only) until P(consistent) >= 0.8, then the fused SAM3 verdict over the nearest views (with the metric-size veto).

## ApexNav adapter (`planners/ApexNav/habitat_evaluation.py`)

- `build_zson_step`: one `ZSONStep` per run from the `verification:` config (needs `sync_step`, the planner launched with
  `verification_gate:=true verification_approach:=true sync_step:=true`; `scripts/run_gate_e2e.sh` does).
- `execute_step`: `zson.observe` with the frame and `target_seen` (ApexNav's detector reported a label-0 detection) after every step; its masks are not passed. The clusters of the reopened
  objects, and the GOTO cluster when `zson.approach_ready()` names its object, ride in `StepObservation.reopen_cluster_ids` (the
  planner forgets a rejection, or ends its native approach and re-requests).
- `run_episode` / `answer_request`: a `VerificationRequest` (cluster centroid) becomes `zson.check(step, centroid,
  final=at_stop, reporting=approach_id > 0, stalled=arrived / failed from viewpoint_status 1 / 2, 3, meta={cluster id, ...})`:

  | verdict | `VerificationResult` |
  |---|---|
  | ACCEPT / REJECT / DEFER | state 1 / -1 / 2 |
  | GOTO | state 3 with `{approach_id, object, camera, max_steps}`: ApexNav's object navigation approaches |
  | CONTROL | `run_zson_control`: level the camera, `zson.act` through `execute_step(external_approach=True)`, then state 1 / -1 / 2 with `resume_step` |

  The cluster id <-> object mapping lives in the adapter only (`meta` is echoed as `cluster` in `decisions.jsonl`; the planner's
  records and `nav.eval.closer_attrib` read it there).
- `NAV_SPLIT` (default true with the step): `run_split` builds the target from the `StepAction` and calls
  `SplitNav.act(..., zson.accepted_points(), body_map)`; sliding must be off. `NAV_SPLIT=false` is the no-split ablation.
- Runs: `planners/ApexNav/scripts/run_pool.sh <episodes> <arm>=<run>` with arms `zson` (step + split), `zson_nosplit` (step
  only) and `baseline` (raw ApexNav, imports nothing from this repo). Outputs in `planners/ApexNav/videos/<run>`; score with
  `python -m nav.eval.closer_attrib` / `nav.eval.nav_table`.

## Adapter checklist (VLFM, `planners/VLFM`, the upstream fork, untouched so far)

VLFM has no object clusters: its target is a point (best frontier / detected object position). That is why the step has its own
object memory.

1. Build one `ZSONStep` with the camera intrinsics, and call `start_episode` (with the simulator for the local planner).
2. After every executed step call `observe` with the RGB, the metric depth, the agent pose `(x, y, yaw)` in VLFM's world
   frame and the camera pitch and `target_seen` = VLFM's detector sees the target (no masks: the step runs SAM3 itself).
3. When VLFM is about to commit to / stop at a detected target at world point `p`, call `check(step, p)` (`final=True` for the
   stop itself). Hold VLFM's action until the verdict:
   - ACCEPT: stop / go on; REJECT or DEFER: drop the candidate (VLFM keeps exploring; ignore `observe`'s return value, or use it to
     un-blacklist a reopened point);
   - GOTO: navigate toward `verdict.point` with VLFM's own navigation; after each step `observe`, then when
     `approach_ready()` or the navigation ended call `check(step, p, reporting=True, stalled=ended)`;
   - CONTROL: level the camera, then each step execute `act(...)`'s action and `observe`, until `act` returns a Verdict.
4. Optionally call `SplitNav.act` before each motion action once VLFM heads for an accepted target.
5. Provide `scene.agent_pose` / `scene.camera` for the simulator in use.

Tests: `python -m unittest discover -s nav -t .` (core env, `PYTHONPATH` = repo root).
