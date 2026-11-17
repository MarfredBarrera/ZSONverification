"""LiveApproach and its identity / decision rules with fakes (no ROS, no GPU):

    python -m unittest nav.gate.test_approach
"""
import math
import unittest

import numpy as np

RGB = np.full((48, 64, 3), 120, dtype=np.uint8)
OBJECT = (2.0, 0.0)


def mask(size):
    m = np.zeros((48, 64), dtype=np.uint8)
    m[24 - size // 2:24 + size // 2, 32 - size // 2:32 + size // 2] = 1
    return m


class LiveFrameTest(unittest.TestCase):
    """LiveApproach._object_xz: planner-frame object point -> sim (x, z), whatever the two frames' offset."""

    def test_object_xz(self):
        from nav.approach import geom as G
        from nav.gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.G = G
        rng = np.random.default_rng(0)
        for _ in range(20):
            psi, yaw = rng.uniform(-math.pi, math.pi, 2)          # the agent's yaw in each frame
            pose_pf = np.array([*rng.uniform(-5, 5, 2), psi])
            position = np.array([rng.uniform(-5, 5), 0.1, rng.uniform(-5, 5)])
            ahead, left = rng.uniform(-3, 3, 2)
            # planner frame (ROS): x along yaw psi, left = +90 deg
            live.object_pf = pose_pf[:2] + ahead * np.array([math.cos(psi), math.sin(psi)]) \
                + left * np.array([-math.sin(psi), math.cos(psi)])
            want = position[[0, 2]] + ahead * G.forward(yaw) + left * G.forward(yaw + math.pi / 2)
            np.testing.assert_allclose(live._object_xz(position, yaw, pose_pf), want, atol=1e-9)
        # sim left of yaw 0 (looking along -z) is -x
        np.testing.assert_allclose(G.forward(math.pi / 2), [-1.0, 0.0], atol=1e-12)

    def test_goto_turns_toward_a_goal_behind(self):
        from nav.approach import geom as G
        from nav.approach import local_nav as L
        from nav.approach import scene as S
        from nav.gate.approach import LiveApproach
        pos = np.array([1.0, 0.1, -2.0])
        live = LiveApproach.__new__(LiveApproach)
        live.G, live.S = G, S
        live.lmap = L.LocalMap(G.HeightMap(pos[[0, 2]], pos[1]))
        live.pf = L.LocalPathfinder(live.lmap, pos[1])
        live.pf.refresh(pos)
        live.last_action, live.last_pos, live.goto_stall = None, None, 0
        here = pos[[0, 2]]
        for yaw in (0.0, 2.0):
            self.assertEqual(live._goto(pos, yaw, here + 4.0 * G.forward(yaw)), "move_forward")
            self.assertIn(live._goto(pos, yaw, here + 4.0 * G.forward(yaw + 2.8)), ("turn_left", "turn_right"))
            self.assertEqual(live._goto(pos, yaw, here + 4.0 * G.forward(yaw + 1.2)), "turn_left")


class LiveSeedTest(unittest.TestCase):
    """LiveApproach._seed: the gate's bank views are named and counted through the diversity gate, never commit."""

    def test_seed_counts_distinct_views_only(self):
        from nav.approach import decide as D
        from nav.approach import geom as G
        from nav.gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.D, live.G, live.target, live.log = D, G, "potted plant", None
        live.P = {**D.DEFAULTS, "GATE": "replace"}
        live.counted, live.belief, live.result, live.seen = [], None, None, []
        names = []

        def name(rgb, m, sam3):
            names.append(sam3)
            return {"options": ["lamp", "potted plant"], "probabilities": {"lamp": 0.05, "potted plant": 0.95},
                    "sam3": {"scores": {"potted plant": 0.8, "lamp": 0.1}}}
        live._name = name
        big = np.zeros((48, 64), np.uint8)
        big[4:44, 4:60] = 1                                  # 2240 px >= 400
        live.bank = [{"step": 3, "index": 0, "rgb": RGB, "mask": big, "camera": (0.0, 0.0), "centroid": (2.0, 0.0)},
                     {"step": 4, "index": 0, "rgb": RGB, "mask": big, "camera": (0.1, 0.0), "centroid": (2.0, 0.0)},
                     {"step": 9, "index": 0, "rgb": RGB, "mask": big, "camera": (2.0, 2.0), "centroid": (2.0, 0.0)},
                     {"step": 10, "index": 0, "rgb": RGB, "mask": mask(8), "camera": (2.0, -2.0), "centroid": (2.0, 0.0)}]
        pose = np.array([0.0, 0.1, 0.0])
        live._seed(pose, 0.0, (0.0, 0.0, 0.0))
        self.assertEqual(len(names), 3)                      # the 64 px view is not usable
        self.assertEqual(len(live.counted), 2)               # step 4 repeats step 3 (0.1 m, same bearing)
        self.assertEqual(sorted(f["step"] for f in live.counted), [-4, -2])
        self.assertIsNone(live.result)
        self.assertEqual(len(live.seen), 3)                  # SAM3 verifies over every usable view
        self.assertIsNotNone(live.belief)


def box(r0, r1, c0, c1, shape=(48, 64)):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


class AssociateTest(unittest.TestCase):
    """approach.associate: the one rule (IoU >= ASSOC_MIN) matching detector masks (ApexNav's or SAM3's) to the
    tracked outline T."""

    def test_sconce_above_orchid_rejected(self):          # 693: SAM3's only candidate is the lamp above the orchid
        from nav.gate.approach import associate
        orchid, sconce = box(28, 44, 26, 38), box(4, 14, 28, 36)
        best, scores = associate([sconce], orchid)
        self.assertIsNone(best)
        self.assertEqual(scores[0][1], 0.0)

    def test_curtain_behind_pedestal_rejected(self):      # 259: the curtain (0.84) beside / behind the pedestal
        from nav.gate.approach import associate
        pedestal = box(20, 46, 26, 36)
        curtain = box(0, 48, 0, 26) | box(0, 18, 26, 64) | box(0, 48, 36, 64)   # around it, not over it
        best, scores = associate([curtain], pedestal)
        self.assertIsNone(best)
        self.assertLess(scores[0][1], 0.3)

    def test_whole_bed_around_bench_rejected(self):       # 304: the outline is the bench at the bed foot (GT couch)
        from nav.gate.approach import associate
        bench, bed = box(36, 46, 14, 50), box(4, 46, 4, 60)
        best, scores = associate([bed], bench)
        self.assertIsNone(best)
        self.assertEqual(scores[0][0], 1.0)               # the overlap coefficient would have accepted it
        self.assertLess(scores[0][1], 0.3)

    def test_same_size_object_accepted(self):
        from nav.gate.approach import associate
        chair = box(10, 40, 20, 44)
        best, scores = associate([box(0, 8, 0, 64), box(11, 41, 21, 45)], chair)
        self.assertEqual(best, 1)
        self.assertGreater(scores[1][1], 0.8)

    def test_tiny_fragment_rejected_whole_object_wins(self):   # a 487 px chair fragment vs the 8.5k whole chair
        from nav.gate.approach import associate
        chair, fragment = box(10, 40, 20, 44), box(12, 16, 22, 26)
        best, scores = associate([fragment], chair)
        self.assertIsNone(best)
        self.assertEqual(scores[0][0], 1.0)
        best, _ = associate([fragment, box(10, 40, 20, 44)], chair)
        self.assertEqual(best, 1)

    def test_ties_by_detector_score_and_empty_inputs(self):
        from nav.gate.approach import associate
        t = box(10, 30, 10, 30)
        self.assertEqual(associate([t, t], t)[0], 0)                         # no detector scores: the first
        self.assertEqual(associate([t, t], t, det_scores=[0.4, 0.9])[0], 1)  # else the higher score
        self.assertEqual(associate([], t), (None, []))
        self.assertIsNone(associate([t], np.zeros((48, 64), bool))[0])   # no outline: no detection


class LiveIdentityTest(unittest.TestCase):
    """LiveApproach._reference / _detect: the seed mask before O exists, SAM3 candidates through associate."""

    def _live(self):
        from nav.gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.cloud, live.object_pf, live.target, live.tag, live.actions, live.url = \
            np.zeros((0, 3)), np.array([2.0, 0.0]), "potted plant", "t", 0, "http://x"
        return live

    def test_seed_is_nearest_centroid_within_seed_m(self):
        live = self._live()
        a, b = box(5, 15, 5, 15), box(30, 40, 30, 40)
        ref, source = live._reference(np.ones((48, 64)), [a, b], [(2.6, 0.0), (2.1, 0.1)])
        self.assertEqual(source, "seed")
        self.assertTrue((ref == b).all())
        ref, source = live._reference(np.ones((48, 64)), [a], [(4.0, 0.0)])
        self.assertEqual((source, int(ref.sum())), ("none", 0))

    def test_detect_takes_overlapping_candidate_only(self):
        import base64
        import cv2
        from unittest import mock
        live = self._live()
        orchid, sconce = box(28, 44, 26, 38), box(4, 14, 28, 36)

        def cand(m, word, score):
            return {"mask": base64.b64encode(cv2.imencode(".png", m.astype(np.uint8) * 255)[1]).decode(),
                    "word": word, "score": score, "scores": {word: score}}
        reply = {"ok": True, "candidates": [cand(sconce, "lamp", 0.9), cand(orchid[:, :] & box(30, 44, 0, 64),
                                                                            "potted plant", 0.6)]}
        with mock.patch("nav.gate.client._call", return_value=reply):
            det, log = live._detect(RGB, orchid)
            self.assertEqual(det["word"], "potted plant")
            self.assertEqual([round(x["overlap"], 2) for x in log], [0.0, 1.0])
            reply["candidates"] = reply["candidates"][:1]
            det, log = live._detect(RGB, orchid)
            self.assertIsNone(det)                        # the sconce alone: no detection this frame

    def test_detect_seeds_by_nearest_centre_without_outline(self):
        """No ApexNav seed and no O yet: the SAM3 candidate whose 3D centre is nearest the object point seeds."""
        import base64
        import types
        import cv2
        from unittest import mock
        from nav.approach import geom as G
        live = self._live()
        live.sim, live.G = None, G
        # fake camera: a mask's points sit at x = its mean column / 10, z = 0
        live.S = types.SimpleNamespace(
            camera=lambda sim: (None, None),
            backproject=lambda d, m, *cam: np.array([[np.nonzero(m)[1].mean() / 10, 0.0, 0.0]]) if m.any()
            else np.zeros((0, 3)))
        far, near = box(10, 20, 2, 8), box(10, 20, 18, 24)        # centres x = 0.45 m and 2.05 m

        def cand(m, word):
            return {"mask": base64.b64encode(cv2.imencode(".png", m.astype(np.uint8) * 255)[1]).decode(),
                    "word": word, "score": 0.8, "scores": {word: 0.8}}
        reply = {"ok": True, "candidates": [cand(far, "lamp"), cand(near, "potted plant")]}
        empty = np.zeros((48, 64), bool)
        with mock.patch("nav.gate.client._call", return_value=reply):
            det, log = live._detect(RGB, empty, np.ones((48, 64), np.float32), np.array([2.0, 0.0]))
            self.assertEqual(det["word"], "potted plant")
            self.assertEqual([x["seed_dist"] for x in log], [1.55, 0.05])
            det, _ = live._detect(RGB, empty, np.ones((48, 64), np.float32), np.array([5.0, 0.0]))
            self.assertIsNone(det)                        # nothing within SEED_M


class NoEvidenceTest(unittest.TestCase):
    """No evidence is no verdict: an approach that never finds the object, or counts no frame, is undecided, and the
    gate defers the cluster (re-check rule) instead of rejecting it."""

    def _live(self, object_xz, yaw=0.0):
        import types
        from nav.approach import geom as G
        from nav.gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        position = np.array([0.0, 0.1, 0.0])
        noop = types.SimpleNamespace(update=lambda *a: None, set_object=lambda *a: None, refresh=lambda *a: None)
        from nav.approach import decide as D
        from nav.approach import size_opinion as SZ
        live.D, live.G, live.SZ, live.sim, live.log, live.target = D, G, SZ, None, None, "tv"
        live.P, live.size_views, live.size_info = dict(D.DEFAULTS), [], {}
        live.S = types.SimpleNamespace(agent_pose=lambda sim: (position.copy(), yaw, 0.0),
                                       camera=lambda sim: (None, None))
        live.hmap = live.lmap = live.pf = noop
        live.cloud, live.floor_y, live.counted, live.seen, live.belief = np.zeros((0, 3)), 0.1, [], [], None
        live.result, live.last_action, live.last_pos, live.stuck = None, None, None, 0
        live.started, live.seeded, live.goto, live.budget = False, True, False, 40
        live.actions = live.face_turns = 0
        live._object_xz = lambda *a: np.asarray(object_xz, float)
        live._detect = lambda *a: (None, [])           # SAM3 finds nothing
        return live

    def _step(self, live):
        return live.step(RGB, np.ones((48, 64), np.float32), [], [], (0.0, 0.0, 0.0))

    def test_facing_and_nothing_seen_is_undecided(self):
        from nav.approach import geom as G
        live = self._live(1.0 * G.forward(0.2))      # 11 deg off: facing
        self.assertIsNone(self._step(live))
        self.assertEqual((live.result["status"], live.result["by"], live.result["end"], live.result["accept"]),
                         ("undecided", "not_found", "facing", False))

    def test_not_facing_turns_then_gives_up_undecided(self):
        from nav.approach import geom as G
        live = self._live(1.0 * G.forward(math.pi / 2))   # 90 deg to the left
        self.assertEqual(self._step(live), "turn_left")
        live.face_turns = 12                              # FACE_TURNS used up
        self.assertIsNone(self._step(live))
        self.assertEqual((live.result["status"], live.result["by"]), ("undecided", "not_found"))

    def test_forced_without_counted_frames_is_undecided(self):
        live = self._live((1.0, 0.0))
        live._forced("planner_done")
        self.assertEqual((live.result["status"], live.result["by"], live.result["accept"]),
                         ("undecided", "empty", False))

    def test_meta_carries_p_consistent(self):
        from nav.approach import decide as D
        live = self._live((1.0, 0.0))
        live.D, live.meta = D, {"target": "tv"}
        self.assertEqual(live._meta(), {"target": "tv", "p_consistent": 0.0})   # no belief yet: no evidence
        live.belief = type("B", (), {"alpha": np.array([9.0, 1.0])})()
        self.assertAlmostEqual(live._meta()["p_consistent"], D.p_consistent(live.belief))


class StopDeltaTest(unittest.TestCase):
    """mppi.MppiV15: the stop tolerance DELTA_eff = DELTA (1 - P_STAR) / (1 - p), clipped to [0.2, 3] DELTA."""

    def test_formula(self):
        from nav.approach.mppi import MppiV15
        from nav.approach import decide as D
        m = MppiV15.__new__(MppiV15)
        d = MppiV15.DELTA
        self.assertEqual(MppiV15.P_STAR, D.DEFAULTS["P_STAR"])
        self.assertEqual(m.stop_delta(None), d)                      # offline harness: no belief key
        self.assertAlmostEqual(m.stop_delta(0.8), d)                 # at P_STAR: today's DELTA
        self.assertAlmostEqual(m.stop_delta(0.5), d * 0.2 / 0.5)
        self.assertAlmostEqual(m.stop_delta(0.9), 2 * d)
        self.assertAlmostEqual(m.stop_delta(0.0), 0.2 * d)           # no evidence yet
        self.assertAlmostEqual(m.stop_delta(0.999), 3 * d)           # clipped
        self.assertAlmostEqual(m.stop_delta(1.0), 3 * d)

    def test_execute_uses_delta_eff(self):
        import types
        from nav.approach.mppi import MppiV15
        m = MppiV15.__new__(MppiV15)
        m._ws_seq, m.warm_info, m._w = None, [], 1.0
        m.top_cut = m.top_seen = m.bot_cut = m.bot_seen = False
        r = {"rollout": np.array([0] * 4), "diff": 0.0, "c0": 1.0, "cost": 1.0, "stay_cost": 1.05, "t*": 3,
             "info": None}

        def obs(meta):
            return types.SimpleNamespace(position=np.zeros(3), yaw=0.0, meta=meta)
        m.done = False
        self.assertIsNone(m.execute(obs({}), r))                     # 0.05 < DELTA 0.1: stop at the best view
        self.assertNotIn("delta", m.last)
        m.done = False
        self.assertEqual(m.execute(obs({"p_consistent": 0.0}), r), "move_forward")   # 0.05 >= 0.02: keep going
        self.assertEqual((m.last["p"], m.last["delta"]), (0.0, 0.02))


class ObjectPointsTest(unittest.TestCase):
    """LiveApproach.object_points: the deciding (nearest verified) view's SAM3 instance, else O."""

    def test_nearest_verified_view_wins(self):
        from nav.approach import decide as D
        from nav.gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.D, live.P, live.cloud = D, dict(D.DEFAULTS), np.zeros((50, 3))

        def frame(step, dist):
            return {"step": step, "pose": {"distance": dist}, "mask": {"area": 0.2}}
        live.seen = [frame(0, 2.0), frame(1, 1.0), frame(2, 1.5)]
        live.view_pts = {0: np.ones((5, 3)), 2: np.full((4, 3), 2.0)}
        self.assertEqual(len(live.object_points()), 4)       # step 1 is nearest but has no instance: step 2
        live.view_pts = {}
        self.assertEqual(len(live.object_points()), 50)      # no instance: O
        live.seen = []
        self.assertEqual(len(live.object_points()), 50)


class SizeVetoTest(unittest.TestCase):
    """The couch size opinion is a veto (decide.fused_accept, size_opinion.likelihood): a complete small width
    rules the couch out, a large complete width (e.g. inflated by a merge, test2 ep 181) never confirms it."""

    def fused(self, belief, accept, width):
        from unittest import mock
        from nav.approach import decide as D
        info = {}
        with mock.patch.object(D.SEL, "fuse_views", return_value={"accept": accept, "belief": belief}):
            out = D.fused_accept([{}], "couch", dict(D.DEFAULTS), {"W": width, "complete": True}, info)
        self.assertEqual(info["accept_after"], out)
        return out, info

    def test_large_width_never_accepts(self):
        # ep 181: chair narrowly ahead of couch; any couch-like or larger width must leave the reject
        belief = {"chair": 0.0965, "couch": 0.0957, "bed": 0.0004, "lamp": 0.0086, "tv": 0.0002}
        for width in (1.0, 1.2, 1.3, 1.5, 1.8, 2.1, 2.9, 4.0, 6.0):
            out, info = self.fused(belief, False, width)
            self.assertFalse(out, width)
            self.assertFalse(info["accept_before"])

    def test_large_width_never_lowers(self):
        # a real couch with a bed-sized or huge width keeps its accept (only "too small" carries evidence)
        belief = {"chair": 0.30, "couch": 0.32, "bed": 0.31, "tv": 0.07}
        for width in (1.5, 1.8, 2.1, 2.9, 4.0):
            out, info = self.fused(belief, True, width)
            self.assertTrue(out, width)
            self.assertFalse(info["applied"], width)

    def test_small_width_vetoes(self):
        belief = {"chair": 0.30, "couch": 0.40, "bed": 0.05, "tv": 0.05}
        out, info = self.fused(belief, True, 0.85)   # an armchair
        self.assertTrue(info["accept_before"])
        self.assertFalse(out)
        self.assertTrue(info["applied"])
        self.assertEqual(info["likelihood"]["couch"], 1.0)
        self.assertTrue(all(v >= 1.0 for v in info["likelihood"].values()))


if __name__ == "__main__":
    unittest.main()
