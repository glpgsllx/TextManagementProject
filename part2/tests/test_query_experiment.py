"""Structural checks on the experiment configuration itself (no full corpus run — this
only validates the experimental design).

A full run takes about a minute, so misconfigured conditions should not have to wait for
that to surface. Three things are pinned down here: condition names must not collide,
every group references conditions that really exist, and each group differs in exactly
one respect (only same-family conditions qualify as a "one change" ablation; the one
deliberate exception is E1, which crosses families).
"""

import unittest

from part2.query_experiment import base_conditions, grid_conditions, group_pairs


class ConditionTests(unittest.TestCase):
    def test_base_conditions_are_six_unique_names(self):
        names = [condition["name"] for condition in base_conditions()]
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)

    def test_every_condition_explains_itself(self):
        for condition in base_conditions() + grid_conditions():
            self.assertTrue(condition["label"].strip())
            self.assertTrue(condition["note"].strip())

    def test_grid_has_sixteen_combinations_and_one_default(self):
        grid = grid_conditions()
        self.assertEqual(len(grid), 16)
        defaults = [condition for condition in grid if condition["is_default"]]
        self.assertEqual(len(defaults), 1)
        self.assertEqual((defaults[0]["k1"], defaults[0]["b"]), (1.2, 0.75))

    def test_grid_names_are_unique(self):
        names = [condition["name"] for condition in grid_conditions()]
        self.assertEqual(len(set(names)), len(names))


class GroupPairTests(unittest.TestCase):
    def test_group_ids_are_unique(self):
        groups = [group for group, _a, _b, _note in group_pairs()]
        self.assertEqual(len(groups), 6)
        self.assertEqual(len(set(groups)), 6)

    def test_group_pairs_reference_existing_conditions(self):
        known = {condition["name"] for condition in base_conditions()}
        for group, name_a, name_b, note in group_pairs():
            self.assertIn(name_a, known, f"{group}: unknown condition {name_a}")
            self.assertIn(name_b, known, f"{group}: unknown condition {name_b}")
            self.assertNotEqual(name_a, name_b)
            self.assertTrue(note.strip())

    def test_only_the_main_comparison_crosses_families(self):
        """Apart from E1 (a head-on comparison of the two implementations), every
        group must be an ablation inside one family."""
        families = {condition["name"]: condition["family"]
                    for condition in base_conditions()}
        for group, name_a, name_b, _note in group_pairs():
            if group == "E1":
                self.assertNotEqual(families[name_a], families[name_b])
            else:
                self.assertEqual(families[name_a], families[name_b],
                                 f"{group}: not a one-change ablation")


if __name__ == "__main__":
    unittest.main()
