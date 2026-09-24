import json
import tempfile
import unittest

from factorgrad import FactorGradController


class FactorGradControllerTest(unittest.TestCase):
    def test_late_schedule_remains_the_default(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=15_000,
            stabilization_steps=768,
        )
        self.assertEqual(controller.decision(3_000).mode, "full_rgb")
        self.assertEqual(controller.decision(15_767).mode, "full_rgb")

    def test_full_schedule_boundaries(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=15_000,
            stabilization_steps=768,
            frozen_probability=2.0 / 3.0,
            schedule="full",
            early_start_iteration=3_000,
        )
        self.assertEqual(controller.decision(2_999).mode, "full_rgb")
        early = controller.decision(3_000)
        self.assertEqual(early.mode, "factorized_channel")
        self.assertEqual(early.phase, "topology_active")
        self.assertEqual(early.temporal_probability, 1.0)
        self.assertEqual(early.temporal_weight, 1.0)
        self.assertEqual(controller.decision(14_999).mode, "factorized_channel")
        self.assertEqual(controller.decision(15_000).mode, "full_rgb")
        self.assertEqual(controller.decision(15_767).mode, "full_rgb")
        late = [controller.decision(15_768 + offset) for offset in range(3)]
        self.assertEqual(
            sum(item.mode == "factorized_channel" for item in late),
            2,
        )

    def test_full_schedule_early_channels_are_stratified(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=15_000,
            schedule="full",
            early_start_iteration=3_000,
        )
        channels = []
        for iteration in range(3_000, 3_006):
            decision = controller.decision(iteration)
            channels.append(decision.selected_channel)
            controller.record(decision, did_backward=True)
        self.assertEqual(set(channels[:3]), {0, 1, 2})
        self.assertEqual(set(channels[3:]), {0, 1, 2})

    def test_exact_temporal_budget_per_block(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=10,
            stabilization_steps=2,
            frozen_probability=2.0 / 3.0,
            seed=7,
        )
        for block in range(12):
            start = controller.freeze_iteration + 3 * block
            decisions = [controller.decision(start + offset) for offset in range(3)]
            self.assertEqual(
                sum(decision.mode == "factorized_channel" for decision in decisions),
                2,
            )

    def test_channel_cycles_are_permutations(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=0,
            stabilization_steps=0,
            frozen_probability=1.0,
            seed=19,
        )
        channels = []
        for iteration in range(12):
            decision = controller.decision(iteration)
            channels.append(decision.selected_channel)
            controller.record(decision, did_backward=True)
        for start in range(0, 12, 3):
            self.assertEqual(set(channels[start : start + 3]), {0, 1, 2})

    def test_rejected_selected_update_does_not_advance_channel(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=0,
            stabilization_steps=0,
            frozen_probability=1.0,
            seed=23,
        )
        first = controller.decision(0)
        controller.record(first, did_backward=False)
        second = controller.decision(1)
        self.assertEqual(first.selected_channel, second.selected_channel)

    def test_without_channel_sampling_is_time_only(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=10,
            stabilization_steps=2,
            frozen_probability=2.0 / 3.0,
            schedule="full",
            early_start_iteration=3,
            channel_sampling=False,
        )
        self.assertEqual(controller.decision(3).mode, "full_rgb")
        decisions = [controller.decision(12 + offset) for offset in range(3)]
        self.assertEqual(
            sum(item.mode == "reweighted_full_rgb" for item in decisions), 2
        )
        self.assertEqual(sum(item.mode == "l1_only" for item in decisions), 1)

    def test_without_temporal_sampling_is_channel_only(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=10,
            stabilization_steps=2,
            frozen_probability=2.0 / 3.0,
            schedule="full",
            early_start_iteration=3,
            temporal_sampling=False,
        )
        decisions = [controller.decision(12 + offset) for offset in range(6)]
        self.assertTrue(all(item.mode == "factorized_channel" for item in decisions))
        self.assertTrue(all(item.temporal_probability == 1.0 for item in decisions))

    def test_without_topology_protection_applies_joint_budget_early(self):
        controller = FactorGradController(
            enabled=True,
            topology_end_iteration=10,
            stabilization_steps=2,
            frozen_probability=2.0 / 3.0,
            schedule="full",
            early_start_iteration=3,
            topology_protection=False,
        )
        self.assertEqual(controller.decision(2).mode, "full_rgb")
        decisions = [controller.decision(3 + offset) for offset in range(3)]
        self.assertEqual(
            sum(item.mode == "factorized_channel" for item in decisions), 2
        )
        self.assertEqual(sum(item.mode == "l1_only" for item in decisions), 1)
        self.assertNotEqual(controller.decision(10).phase, "post_topology_stabilization")

    def test_save_writes_machine_readable_summary(self):
        controller = FactorGradController(enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            path = controller.save(directory)
            with open(path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        self.assertEqual(summary["method"], "FactorGrad-GS")


if __name__ == "__main__":
    unittest.main()
