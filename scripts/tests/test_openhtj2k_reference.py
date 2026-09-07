# SPDX-License-Identifier: MIT OR Apache-2.0

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare-openhtj2k-reference.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "full-validation.yml"


class OpenHtj2kReferenceTests(unittest.TestCase):
    def test_prepare_script_pins_the_official_source_and_parses_as_bash(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("https://github.com/osamu620/OpenHTJ2K.git", source)
        self.assertIn("e0f7ae853220d1e359c438b0bb6ad6cb2b3899db", source)
        self.assertIn('version="0.19.0"', source)
        self.assertIn("J2K_OPENHTJ2K_DEC_BIN", source)
        self.assertIn("J2K_OPENHTJ2K_SOURCE_DIR", source)
        self.assertIn("J2K_OPENHTJ2K_LIB_DIR", source)
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_reference_uses_the_shims_dynamic_msvc_runtime(self):
        source = SCRIPT.read_text(encoding="utf-8")
        configure = source.split("\ncmake \\\n", 2)[1]
        # OpenHTJ2K replaces CMAKE_CXX_FLAGS_RELEASE. Runtime selection must
        # survive that replacement and match the Rust cc shim's /MD runtime.
        self.assertIn("-DCMAKE_POLICY_DEFAULT_CMP0091=NEW", configure)
        self.assertIn("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL", configure)

    def test_cpu_evidence_lanes_prepare_the_reference_before_running_t803(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        job = workflow.split("  t803-cpu:\n", 1)[1].split("\n  metal-compile:\n", 1)[0]

        prepare = "scripts/prepare-openhtj2k-reference.sh"
        run = "cargo xtask t803 run --iut cpu"
        self.assertIn(prepare, job)
        self.assertIn(run, job)
        self.assertLess(job.index(prepare), job.index(run))


if __name__ == "__main__":
    unittest.main()
