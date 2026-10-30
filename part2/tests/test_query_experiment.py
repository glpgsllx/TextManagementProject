"""实验配置本身的结构核对（不跑全量语料，只验实验设计）。

跑一次全量要 1 分钟，所以"实验组有没有配错"这种事不该等到跑完才发现。
这里钉住三条：条件名不撞车、每组引用的条件确实存在、每组的差异真的只有一处
（两条件的 family 相同才谈得上"只改一处"，跨 family 的只有 E1 是故意的）。
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
            self.assertIn(name_a, known, f"{group} 引用了不存在的条件 {name_a}")
            self.assertIn(name_b, known, f"{group} 引用了不存在的条件 {name_b}")
            self.assertNotEqual(name_a, name_b)
            self.assertTrue(note.strip())

    def test_only_the_main_comparison_crosses_families(self):
        """除 E1（两个实现的正面比较）外，每组都必须是同族消融。"""
        families = {condition["name"]: condition["family"]
                    for condition in base_conditions()}
        for group, name_a, name_b, _note in group_pairs():
            if group == "E1":
                self.assertNotEqual(families[name_a], families[name_b])
            else:
                self.assertEqual(families[name_a], families[name_b],
                                 f"{group} 跨了族，就不是「只改一处」的消融了")


if __name__ == "__main__":
    unittest.main()
