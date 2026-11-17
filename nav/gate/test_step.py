"""ZSONStep with a fake server and a fake LiveApproach (no ROS, no GPU):

    python -m unittest nav.gate.test_step
"""
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from nav.gate.step import ACCEPT, CONTROL, DEFER, GOTO, REJECT, ZSONStep, target_name

H, W = 48, 64
RGB = np.full((H, W, 3), 120, dtype=np.uint8)
DEPTH = np.full((H, W), 2.0, np.float32)     # a wall 2 m ahead: a centred mask has its centroid 2 m ahead of the agent


def mask(size=16):
    m = np.zeros((H, W), dtype=np.uint8)
    m[H // 2 - size // 2:H // 2 + size // 2, W // 2 - size // 2:W // 2 + size // 2] = 1
    return m


class StepTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.statuses = [], []
        self.folder = tempfile.mkdtemp()

        def verify(url, rgb, m, target, input_elevation=0.0, request_id=None, timeout=None):
            self.calls.append(request_id)
            status = self.statuses.pop(0)
            return {"ok": True, "view_image": None,
                    "verdict": {"accepted": status == "correct", "label": "potted plant", "status": status,
                                "source": "sam3" if status != "insufficient" else "jev"}}

        self.step = ZSONStep({"url": "http://x"}, {"hfov": 79, "width": W, "height": H}, self.folder, verify=verify,
                             ping=lambda url, timeout: {"ok": True, "settings": {"selector": "jev_sam3"}})
        self.step.start_episode(7, "scene", "ep", "potted_plant", ["42"], 500, sim=None)

    def observe(self, n, pose=(0.0, 0.0, 0.0), size=16, depth=2.0):
        return self.step.observe(n, RGB, np.full((H, W), depth, np.float32), pose, [mask(size)], [0.5],
                                 -math.radians(30), [[{"id": "42", "pixels": 10}]])

    def decisions(self):
        with open(os.path.join(self.folder, "decisions.jsonl"), encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]

    def test_target_name(self):
        self.assertEqual(target_name("potted_plant"), "potted plant")
        self.assertEqual(target_name("table | dining table | desk"), "table")

    def test_centroid_and_association(self):
        self.observe(1)                                   # 2 m ahead of the origin
        self.observe(2, (0.3, 0.0, 0.0))                  # 0.3 m nearer: the same object
        self.observe(3, (0.0, 0.0, math.pi / 2))          # looking along +y: another object at (0, 2)
        self.assertEqual(len(self.step.objects), 2)
        first, second = self.step.objects
        self.assertEqual((first["n"], len(list(self.step.evidence))), (2, 3))
        np.testing.assert_allclose(first["centroid"], (2.15, 0.0), atol=0.03)
        np.testing.assert_allclose(second["centroid"], (0.0, 2.0), atol=0.03)

    def test_verdict_mapping(self):
        for status, expected in (("correct", ACCEPT), ("wrong", REJECT)):
            self.statuses.append(status)
            self.observe(1 + len(self.calls), (0.0, -10.0 * len(self.calls), 0.0))
            point = self.step.objects[-1]["centroid"]
            self.assertEqual(self.step.check(1 + len(self.calls), point).kind, expected)
        counts = self.step.finish_episode()
        self.assertEqual((counts["accepted"], counts["rejected"], counts["sam3_decisions"]), (1, 1, 2))

    def test_judged_view_answers_from_memory(self):
        self.statuses.append("wrong")
        self.observe(1)
        self.assertEqual(self.step.check(1, (2.0, 0.0)).kind, REJECT)
        self.assertEqual(self.step.check(5, (2.5, 0.3)).kind, REJECT)       # the planner triggers again: same view
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.decisions()[-1]["source"], self.step.finish_episode()["cached"]), ("cached", 1))

    def test_recheck_reopens_after_cooldown_with_a_novel_view(self):
        self.statuses += ["wrong", "correct"]
        self.observe(1)
        point = self.step.objects[0]["centroid"]
        self.assertEqual(self.step.check(1, point).kind, REJECT)
        self.assertEqual(self.observe(10, (-1.2, 0.0, 0.0), depth=3.2), [])    # novel view, but inside the cooldown
        self.assertEqual(self.observe(26, (0.0, 0.0, 0.0)), [])               # cooldown over, the same camera: not novel
        reopened = self.observe(27, (-1.2, 0.0, 0.0), depth=3.2)               # 1.2 m from the view used: novel
        self.assertEqual([r["object"] for r in reopened], [0])
        np.testing.assert_allclose(reopened[0]["point"], point, atol=0.1)
        self.assertEqual(self.step.check(28, point).kind, ACCEPT)             # the novel view is judged, not the old one
        self.assertEqual((self.calls[-1].endswith("_s28"), self.step.finish_episode()["reopened"]), (True, 1))

    def test_recheck_stops_at_max_checks(self):
        self.step.cfg["max_checks"] = 1
        self.statuses.append("wrong")
        self.observe(1)
        self.step.check(1, self.step.objects[0]["centroid"])
        self.assertEqual(self.observe(30, (-1.2, 0.0, 0.0), depth=3.2), [])

    def test_no_object_defers_and_final_rejects(self):
        self.assertEqual(self.step.check(1, (2.0, 0.0)).kind, DEFER)
        self.assertEqual(self.step.check(2, (2.0, 0.0), final=True).kind, REJECT)
        self.assertEqual(self.step.finish_episode()["no_evidence"], 2)
        self.assertEqual(self.calls, [])

    def test_insufficient_at_final_rejects(self):
        self.statuses.append("insufficient")
        self.observe(1)
        self.assertEqual(self.step.check(1, (2.0, 0.0), final=True).kind, REJECT)
        self.assertEqual(self.decisions()[-1]["policy"], "insufficient,final")

    def test_goto_then_control(self):
        self.statuses.append("insufficient")
        self.observe(1)                                   # the object 2 m ahead
        point = self.step.objects[0]["centroid"]
        self.step.observe(2, RGB, DEPTH, (-8.0, 0.0, 0.0), [], [], 0.0)       # the agent is far away now
        verdict = self.step.check(2, point)
        self.assertEqual((verdict.kind, verdict.approach_id, verdict.camera), (GOTO, 1, (0.0, 0.0)))
        self.assertEqual(self.step.check(4, point, reporting=True).kind, GOTO)          # still far: the planner goes on
        self.observe(5, (0.0, 0.0, 0.0))                   # in view again, but from the same range: not a new view
        self.assertEqual(self.step.approach_ready(), [])
        self.observe(9, (0.8, 0.0, 0.0), depth=1.2)        # 0.8 m closer, in view
        self.assertEqual(self.step.approach_ready(), [0])
        self.assertEqual(self.step.approach_ready(), [])   # once
        self.assertEqual(self.step.check(9, point, reporting=True).kind, CONTROL)
        counts = self.step.finish_episode()
        self.assertEqual((counts["moves"], counts["approach_early"], counts["approach_steps"], counts["requests"]),
                         (1, 1, 7, 1))

    def test_stalled_goto_takes_control_anywhere(self):
        self.statuses.append("insufficient")
        self.observe(1)
        point = self.step.objects[0]["centroid"]
        self.step.observe(2, RGB, DEPTH, (-8.0, 0.0, 0.0), [], [], 0.0)
        self.step.check(2, point)
        self.assertEqual(self.step.check(30, point, reporting=True, stalled="arrived").kind, CONTROL)
        self.assertEqual(self.step.finish_episode()["approach_arrived"], 1)

    def test_close_at_first_check_is_control_and_act_decides(self):
        self.statuses.append("insufficient")
        self.observe(1, (0.0, 0.0, 0.0))                  # 2 m away, in view
        verdict = self.step.check(1, (2.0, 0.0), meta={"id": 6})
        self.assertEqual(verdict.kind, CONTROL)
        live = mock.MagicMock()
        live.step.side_effect = ["turn_left", None]
        live.result = {"accept": True, "status": "correct", "by": "belief", "end": None, "actions": 9, "counted": 3,
                       "views": [0], "seconds": 1.0}
        live.object_points.return_value = np.zeros((5, 3))
        live.log_dir = None
        with mock.patch("nav.gate.approach.LiveApproach", return_value=live) as cls:
            self.assertEqual(self.step.act(3, RGB, DEPTH, (0.0, 0.0, 0.0), [mask()]), "turn_left")
            out = self.step.act(4, RGB, DEPTH, (0.0, 0.0, 0.0), [mask()])
        self.assertEqual(out.kind, ACCEPT)
        self.assertEqual(cls.call_args.kwargs["bank"], self.step.objects[0]["bank"])
        self.assertEqual(len(self.step.accepted_points()), 1)
        self.assertEqual(self.decisions()[-2]["cluster"], {"id": 6})         # the approach's decision keeps the request
        with self.assertRaises(RuntimeError):
            self.step.act(5, RGB, DEPTH, (0.0, 0.0, 0.0), [])

    def test_reporting_without_an_approach_raises(self):
        self.observe(1)
        with self.assertRaises(RuntimeError):
            self.step.check(2, (2.0, 0.0), reporting=True)

    def test_budget(self):
        self.step.cfg["approach_max_per_object"] = 1
        self.statuses += ["insufficient", "insufficient"]
        self.observe(1)
        self.assertEqual(self.step.check(1, (2.0, 0.0)).kind, CONTROL)
        self.step.objects[0]["pending"] = None             # the approach ended without a verdict (deferred)
        self.assertEqual(self.step.check(60, (2.0, 0.0)).kind, REJECT)
        self.assertEqual(self.decisions()[-1]["policy"], "insufficient,object_budget")

    def test_server_error_raises(self):
        def down(*args, **kwargs):
            raise TimeoutError("timed out")
        self.step._verify = down
        self.observe(1)
        with self.assertRaises(TimeoutError):
            self.step.check(1, (2.0, 0.0))
        self.assertIn("TimeoutError", self.decisions()[-1]["error"])

    def test_selector_and_url_are_checked(self):
        with self.assertRaises(ValueError):
            ZSONStep({"url": "http://x"}, {"hfov": 79, "width": W, "height": H}, self.folder,
                     ping=lambda url, timeout: {"settings": {"selector": "vqf"}})
        with self.assertRaises(ValueError):
            ZSONStep({}, {"hfov": 79, "width": W, "height": H}, self.folder)
        with self.assertRaises(ValueError):
            ZSONStep({"url": "http://x", "on_error": "accept"}, {"hfov": 79, "width": W, "height": H}, self.folder)

    def test_evidence_keeps_the_best_and_the_newest_per_cell(self):
        self.step.evidence.per_cell = 2
        for n, size in enumerate((16, 24, 8, 12, 6), start=1):
            self.observe(n, size=size)
        views = list(self.step.evidence)
        self.assertEqual(len(views), 3)
        self.assertIn(5, [v["step"] for v in views])       # the newest stays
        self.assertIn(2, [v["step"] for v in views])       # the most readable stays

    def test_bank_seeded_from_the_evidence_and_grows_while_approaching(self):
        self.statuses.append("insufficient")
        self.observe(1)
        self.observe(2, (0.0, 0.0, 0.0), size=24)
        self.step.observe(3, RGB, DEPTH, (-8.0, 0.0, 0.0), [], [], 0.0)
        verdict = self.step.check(3, self.step.objects[0]["centroid"])
        self.assertEqual(verdict.kind, GOTO)
        self.assertEqual(len(self.step.objects[0]["bank"]), 2)
        self.observe(7, (0.8, 0.0, 0.0), depth=1.2)
        self.assertEqual(len(self.step.objects[0]["bank"]), 3)


class SameObjectTest(unittest.TestCase):
    """A check on an object an approach already decided, seen as another object, inherits the verdict (the identity rule:
    IoU of the target mask with the decided object's depth-tested projection)."""

    def test_same_object_inherits(self):
        from nav.approach import scene as S
        cam = (np.zeros(3), np.eye(3))
        depth = np.full((S.HEIGHT, S.WIDTH), 3.0, np.float32)
        box = np.zeros((S.HEIGHT, S.WIDTH), bool)
        box[S.HEIGHT // 3: 2 * S.HEIGHT // 3, S.WIDTH // 3: S.WIDTH // 2] = True
        depth[box] = 2.0
        step = ZSONStep({"url": "http://x"}, {"hfov": 79, "width": S.WIDTH, "height": S.HEIGHT}, tempfile.mkdtemp(),
                        ping=lambda url, timeout: {"settings": {"selector": "jev_sam3"}})
        step.start_episode(1, "s", "e", "chair", [], 500, sim=None)
        step.decided.append((S.backproject(depth, box, *cam), REJECT))
        step.last = {"depth": depth, "masks": [box.astype(np.uint8)], "centroids": [(2.0, 0.0)],
                     "pose": (0.0, 0.0, 0.0)}
        with mock.patch("nav.approach.scene.camera", return_value=cam):
            self.assertEqual(step.check(5, (2.0, 0.0)).kind, REJECT)
            step.last["centroids"] = [(6.0, 0.0)]                        # the mask is not at the requested point
            self.assertEqual(step.check(6, (2.0, 0.0)).kind, DEFER)
        self.assertEqual(step.finish_episode()["same_object"], 1)


if __name__ == "__main__":
    unittest.main()
