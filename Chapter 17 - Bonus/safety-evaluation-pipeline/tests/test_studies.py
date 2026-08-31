import unittest

from safety_eval.hardware import Accelerator, Machine
from safety_eval.studies import CATALOG_PATH, load_catalog, plan_study


class StudyPlanningTests(unittest.TestCase):
    def machine(self, memory_gib=16, accelerators=()):
        return Machine("1.0", "linux", "x86_64", 8, memory_gib * 1024**3, tuple(accelerators))

    def test_catalog_covers_every_executable_book_chapter(self):
        chapters = {study["chapter"] for study in load_catalog()["studies"]}
        self.assertTrue(set(range(2, 18)).issubset(chapters))

    def test_reference_notebooks_exist(self):
        for study in load_catalog()["studies"]:
            reference = study["reference_executor"]
            if reference.endswith(".ipynb"):
                self.assertTrue((CATALOG_PATH.parent / reference).resolve().exists(), reference)

    def test_cpu_still_gets_reward_hacking_toy_stage(self):
        plan = plan_study("ch10-reward-hacking", machine=self.machine())
        self.assertEqual(plan["recommended_stage"], "toy")
        self.assertEqual(plan["stages"][1]["missing_capabilities"], ["accelerator"])

    def test_high_memory_cuda_gets_confirmatory_stage(self):
        gpu = Accelerator("cuda", "test GPU", 48 * 1024**3)
        plan = plan_study("ch10-reward-hacking", machine=self.machine(64, (gpu,)))
        self.assertEqual(plan["recommended_stage"], "full")


if __name__ == "__main__":
    unittest.main()
