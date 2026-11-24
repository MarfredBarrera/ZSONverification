"""Eye-corrected SPL: success is the eye label S (STOP within 1 m of a real target, annotated or not).

  python -m nav.eval.eye_spl RUN [RUN ...] [--instances TSV] [--out TSV]       (needs habitat_sim + numpy: vlfm or apexnav container)

  eye label != S                      -> SPL 0
  eye S and Habitat success           -> Habitat's spl
  eye S and Habitat failure           -> L* / max(P, L*), P = the executed path (3D, navmesh heights), L* = min(Habitat's
        start->goal geodesic, geodesic start -> nearest view point of the eye-verified object the agent stopped at).
        View points = navigable points on the agent's floor within 1 m (horizontal) of the instance's bounding box.
        The instance ids come from the --instances TSV (run, episode, instances, note), chosen by eye; every such episode needs a row,
        an empty `instances` means the annotated goal is the object reached (L* = annotated), `stop` an object missing from the
        semantic mesh (view points within 1 m of the final position).
Habitat-success episodes also get spl_check = L/max(P, L), the recomputed spl (sanity check of P).
Runs: see RUNS (ApexNav OVON + HM3D-v2, VLFM HM3D-v2). Planner pose (x forward, y left from the start)
maps to Habitat (x, z) through the start pose, as in outputs/apexnav_online/labels/make_evidence.py."""
import argparse, csv, glob, gzip, json, math, os
import numpy as np
import habitat_sim

from nav.eval.closer_attrib import jl
from nav.eval import semantic_objects

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LABELS = ROOT + "/outputs/apexnav_online/labels/"
RUNS = {"ovon_raw200": (ROOT + "/planners/ApexNav/videos/ovon_raw200/baseline", "ovon"),
        "ovon_split200": (ROOT + "/planners/ApexNav/videos/ovon_split200/gate", "ovon"),
        **{r: (f"{ROOT}/planners/ApexNav/videos/{r}/{a}", "hm3d") for r, a in (("hm3d_raw1000", "baseline"), ("hm3d_split1000", "gate"),
                                                                             ("hm3d_split1000_s1", "gate"), ("hm3d_split1000_s2", "gate"))},
        **{r: (f"{ROOT}/planners/VLFM/outputs/{r}/{a}", "hm3d") for r, a in (("hm3d_rec_raw1000", "baseline"), ("hm3d_rec_zson1000", "gate"),
           ("hm3d_rec_zson1000_s1", "gate"), ("hm3d_rec_zson1000_s1b", "gate"), ("hm3d_rec_zson1000_s2", "gate"))}}
DATA = next(p for p in ("/workspace/ApexNav/data", "/workspace/VLFM/data", "/ICONAS/users/marfred/apexnav_data") if os.path.isdir(p))
COLS = ["run", "episode", "eye", "habitat_success", "habitat_spl", "P", "L_annotated", "L_object", "instance", "eye_spl", "spl_check", "note"]


def navmesh(scene):
    """Habitat's own navmesh for the runs' agent (recomputed with radius 0.18, height 0.88, as at run time); returns the Simulator
    (no renderer) that owns it. Reproduces the logged start->goal geodesic exactly."""
    b = f"{DATA}/scene_datasets/hm3d/"
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = f"{b}val/{scene}/{scene.split('-')[1]}.basis.glb"
    cfg.scene_dataset_config_file = b + "hm3d_annotated_basis.scene_dataset_config.json"
    cfg.create_renderer, cfg.requires_textures = False, False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
    ns = habitat_sim.nav.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height, ns.agent_max_climb, ns.agent_max_slope = 0.18, 0.88, 0.2, 45.0
    assert sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def objects(scene):
    """{instance id: (category, Nx3 vertices in the Habitat frame)}; the semantic glb is z-up: habitat = (x, z, -y)."""
    return {i: (n, v[:, [0, 2, 1]] * [1, 1, -1]) for i, (n, v) in semantic_objects.objects(scene, DATA).items()}


def meta_of(sp, E):
    """The record's meta.json; records back-filled into hm3d_raw1000 have none, so its episode row stands in (ApexNav HM3D-v2)."""
    m = os.path.join(os.path.dirname(sp), "meta.json")
    return json.load(open(m)) if os.path.exists(m) else {**E, "dataset": "hm3dv2"}


def start_pose(meta, E):
    if "start_position" in meta:  # VLFM recorder
        return meta["start_position"], meta["start_rotation"]
    v2 = meta.get("dataset") == "hm3dv2"  # ApexNav HM3D-v2 ObjectNav, else OVON val_unseen
    sub = "objectnav/hm3d/v2/val" if v2 else "ovon/hm3d/val_unseen"
    ds = json.load(gzip.open(f"{DATA}/datasets/{sub}/content/{meta['scene'].split('-')[1]}.json.gz"))
    e = ds["episodes"][int(meta["episode_id"])]  # ApexNav's episode_id = index in the scene file
    assert {"sofa": "couch", "tv_monitor": "tv", "plant": "potted plant"}.get(e["object_category"], e["object_category"]) == E["target_label"]
    return e["start_position"], e["start_rotation"]


def path(steps, start, rot, pf):
    """Executed 3D path, (length, final xyz): the logged Habitat positions (VLFM records), else the planner poses mapped to
    Habitat (x, z) with heights from the navmesh."""
    if "habitat_pose" in steps[0]:
        pts = np.array([s["habitat_pose"]["position"] for s in steps])
        return np.linalg.norm(np.diff(pts, axis=0), axis=1).sum(), pts[-1]
    th = 2 * math.atan2(rot[1], rot[3])
    f, l = np.array([-math.sin(th), -math.cos(th)]), np.array([-math.cos(th), math.sin(th)])
    prev, length, island = np.array(start, float), 0.0, pf.get_island(np.array(start, np.float32))
    for s in steps:
        xz = np.array([start[0], start[2]]) + s["pose"][0] * f + s["pose"][1] * l
        q = np.array(pf.snap_point(np.array([xz[0], prev[1], xz[1]], np.float32), island_index=island), float)
        length += np.linalg.norm(q - prev)
        prev = q
    return length, prev


def viewpoint_geodesic(pf, start, final, verts):
    """Geodesic from start to the nearest navigable point (agent's floor) within 1 m horizontally of the vertices' bounding box."""
    lo, hi = verts[:, [0, 2]].min(0), verts[:, [0, 2]].max(0)
    gx, gz = np.meshgrid(np.arange(lo[0] - 1, hi[0] + 1.001, 0.1), np.arange(lo[1] - 1, hi[1] + 1.001, 0.1))
    pts, island = [], pf.get_island(np.array(start, np.float32))
    for x, z in zip(gx.ravel(), gz.ravel()):
        if np.hypot(max(lo[0] - x, 0, x - hi[0]), max(lo[1] - z, 0, z - hi[1])) > 1:
            continue
        p = np.array([x, final[1], z], np.float32)
        if pf.is_navigable(p):
            q = np.array(pf.snap_point(p), np.float32)
            if pf.get_island(q) == island:
                pts.append(q)
    assert pts, "no navigable view point"
    mp = habitat_sim.MultiGoalShortestPath()
    mp.requested_start, mp.requested_ends = np.array(start, np.float32), pts
    assert pf.find_path(mp), f"view points unreachable from the start ({len(pts)} points)"
    return mp.geodesic_distance


def episode(run, ep, label, instances, cache):
    d = f"{RUNS[run][0]}/epi{ep}"
    E = jl(d + "/episodes.jsonl")[-1]
    sp = glob.glob(d + "/records/*/*/steps.jsonl")[0]
    steps, meta = jl(sp), meta_of(sp, E)
    row = dict(run=run, episode=ep, eye=label, habitat_success=int(E["success"]), habitat_spl=E["spl"], eye_spl=0.0)
    if label != "S":
        return row
    if meta["scene"] not in cache:
        for sim in cache.values():
            sim.close()
        cache.clear()
        cache[meta["scene"]] = navmesh(meta["scene"])
    pf = cache[meta["scene"]].pathfinder
    start, rot = start_pose(meta, E)
    P, final = path(steps, start, rot, pf)
    L = steps[0]["distance_to_goal"]  # Habitat's start->goal geodesic (the agent has not moved at step 1)
    row.update(P=round(P, 3), L_annotated=round(L, 3))
    if E["success"]:
        row.update(eye_spl=E["spl"], spl_check=L / max(P, L))
        return row
    inst = instances[(run, ep)]
    row["note"] = inst["note"]
    Ls = L
    if inst["instances"] == "stop":
        Lo = viewpoint_geodesic(pf, start, final, np.array([final]))
        row.update(L_object=round(Lo, 3), instance="stop")
        Ls = min(L, Lo)
    elif inst["instances"]:
        objs = objects(meta["scene"])
        ids = [int(i) for i in inst["instances"].split(",")]
        Lo = viewpoint_geodesic(pf, start, final, np.concatenate([objs[i][1] for i in ids]))
        row.update(L_object=round(Lo, 3), instance=";".join(f"{i}:{objs[i][0]}" for i in ids))
        Ls = min(L, Lo)
    row["eye_spl"] = Ls / max(P, Ls)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", choices=list(RUNS))
    ap.add_argument("--instances", default=LABELS + "eye_spl_instances.tsv")
    ap.add_argument("--out", default=LABELS + "eye_spl.tsv")
    a = ap.parse_args()
    instances = {(r["run"], r["episode"]): r for r in csv.DictReader(open(a.instances), delimiter="\t")}
    ledger = [r for r in csv.DictReader(open(LABELS + "ledger.tsv"), delimiter="\t") if r["verified"] == "eye" and r["run"] in a.runs]
    rows, cache = [], {}
    for r in sorted(ledger, key=lambda r: (r["run"], jl_scene(r), int(r["episode"]))):
        rows.append(episode(r["run"], r["episode"], r["label"], instances, cache))
    with open(a.out, "w") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader(); w.writerows(rows)
    for run in a.runs:
        R = [r for r in rows if r["run"] == run]
        chk = [abs(r["spl_check"] - r["habitat_spl"]) for r in R if "spl_check" in r]
        unres = [r["episode"] for r in R if r["eye"] == "S" and not r["habitat_success"] and not r.get("instance")]  # annotated goal
        print(f"{run}: n={len(R)} eyeS={sum(r['eye'] == 'S' for r in R)} bench_spl={100 * np.mean([r['habitat_spl'] for r in R]):.2f} "
              f"eye_spl={100 * np.mean([r['eye_spl'] for r in R]):.2f} spl_check max|diff|={max(chk):.4f} (n={len(chk)}) unresolved={unres}")


def jl_scene(r):
    d, _ = RUNS[r["run"]]
    return jl(f"{d}/epi{r['episode']}/episodes.jsonl")[-1]["scene"]  # group by scene: one navmesh/semantic load per scene


if __name__ == "__main__":
    main()
