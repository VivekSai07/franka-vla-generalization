"""The arm closure recorded by `prepare` covers everything the X-VLA adapter really loads. Robosuite image, GPU, bundle at /bundle."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    raise unittest.SkipTest("needs a GPU")

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path(os.environ.get("BUNDLE", "/bundle"))
SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from experiments.vla_inputs_eval import adapters, contract, evaluate
ident = "experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter"
declared = set(evaluate.arm_files(ident)) | set(contract.core_provenance())
adapter = adapters.load_adapter(ident, Path(sys.argv[1]))
obs = dict(images=[np.zeros((512, 512, 3), np.uint8)] * 2, state=np.zeros(9), instruction="put the can in the blue bin")
adapter.reset()
adapter.act(obs)
root = contract.ROOT
loaded = set()
for m in list(sys.modules.values()):
    p = getattr(m, "__file__", None)
    if p and Path(p).is_absolute() and Path(p).resolve().is_relative_to(root):
        loaded.add(Path(p).resolve().relative_to(root).as_posix())
print(json.dumps(sorted(x for x in loaded if x not in declared and "/tests/" not in x and not x.endswith("__init__.py"))))
"""

READ_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
from experiments.vla_inputs_eval import adapters, contract
opened = set()
sys.addaudithook(lambda event, args: opened.add(str(args[0])) if event == "open" and isinstance(args[0], (str, bytes)) else None)
ident = "experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter"
checkpoint = Path(sys.argv[1])
adapter = adapters.load_adapter(ident, checkpoint)
adapter.reset()
adapter.act(dict(images=[np.zeros((512, 512, 3), np.uint8)] * 2, state=np.zeros(9), instruction="put the can in the blue bin"))
module = sys.modules["experiments.vla_inputs_arm_xvla.adapter"]
allowed = [str(checkpoint.resolve())] + [str((contract.ROOT / d).resolve()) for d in module.INPUT_DIRS]
watched = (str(contract.ROOT / ".cache"), "/bundle")
read = sorted({p for p in (str(Path(x).resolve()) for x in opened if x.startswith("/")) if p.startswith(watched)
               and not any(p.startswith(a) for a in allowed)})
touched = sorted({p for p in opened if p.startswith(str(contract.ROOT / ".cache"))})
print(json.dumps(dict(unbound=read, touched_cache_files=len(touched))))
"""


class ArmClosure(unittest.TestCase):
    def test_everything_the_adapter_loads_is_in_the_core_or_the_arm_closure(self):
        env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        out = subprocess.run([sys.executable, "-c", SCRIPT, str(BUNDLE / "checkpoints/step_101520")], capture_output=True,
                             text=True, cwd=ROOT, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), [])

    def test_the_adapter_reads_nothing_under_cache_or_bundle_outside_the_checkpoint_and_its_input_directories(self):
        env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        out = subprocess.run([sys.executable, "-c", READ_SCRIPT, str(BUNDLE / "checkpoints/step_101520")], capture_output=True,
                             text=True, cwd=ROOT, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[-1500:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(result["unbound"], [])
        self.assertGreater(result["touched_cache_files"], 0)  # the tokenizer really is read from the declared directory

    def test_the_declared_closure_names_the_adapter_and_its_p2_dependencies(self):
        from experiments.vla_inputs_eval import evaluate
        files = evaluate.arm_files("experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter")
        for rel in ("experiments/vla_inputs_arm_xvla/adapter.py", "experiments/vla_inputs_arm_xvla/README.md",
                    "experiments/xvla_props_p2_eval/evaluate.py", "experiments/xvla_props_p2_eval/protocol.py",
                    "experiments/xvla_single_task/contract.py"):
            self.assertIn(rel, files)


if __name__ == "__main__":
    unittest.main()
