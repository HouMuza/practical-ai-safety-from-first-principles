from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from safety_eval.hardware import Accelerator, Machine, load_profiles, recommend_profile


PROFILES = Path(__file__).parents[1] / "configs" / "execution-profiles.json"


class ProfileRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = load_profiles(PROFILES)

    def machine(self, memory_gib: int, accelerators=()) -> Machine:
        return Machine("1.0", "linux", "x86_64", 8, memory_gib * 1024**3, tuple(accelerators))

    def test_cpu_is_given_smoke_profile(self) -> None:
        self.assertEqual(recommend_profile(self.machine(64), self.profiles)["profile_id"], "smoke_cpu")

    def test_consumer_cuda_is_sampled(self) -> None:
        gpu = Accelerator("cuda", "Example GPU", 12 * 1024**3)
        self.assertEqual(recommend_profile(self.machine(32, [gpu]), self.profiles)["profile_id"], "sampled_accelerator")

    def test_large_cuda_can_run_full_profile(self) -> None:
        gpu = Accelerator("cuda", "Example GPU", 80 * 1024**3)
        self.assertEqual(recommend_profile(self.machine(128, [gpu]), self.profiles)["profile_id"], "full_accelerator")


if __name__ == "__main__":
    unittest.main()
