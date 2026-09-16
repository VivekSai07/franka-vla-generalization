"""Reproduce endpoint dtype/equality and scalar rounding evidence, no optimizer."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unchanged, summary = [], {}
    with safe_open(args.base, framework="pt", device="cpu") as a, safe_open(args.checkpoint, framework="pt", device="cpu") as b:
        for name in b.keys():
            if ".lm_expert." not in name:
                continue
            old, new = a.get_tensor(name), b.get_tensor(name)
            same, dtype = torch.equal(old, new), str(new.dtype)
            count = summary.setdefault(dtype, {"tensors": 0, "unchanged": 0})
            count["tensors"] += 1
            count["unchanged"] += int(same)
            if same:
                unchanged.append({"name": name, "dtype": dtype, "shape": list(new.shape),
                                  "absmax": float(new.abs().max())})
    one = torch.tensor(1., dtype=torch.bfloat16)
    rounding = {"bf16_one": float(one),
                "bf16_next_lower": float(torch.nextafter(one, torch.tensor(0., dtype=torch.bfloat16))),
                "bf16_next_upper": float(torch.nextafter(one, torch.tensor(2., dtype=torch.bfloat16))),
                "bf16_one_minus_1e4": float(one - torch.tensor(1e-4, dtype=torch.bfloat16)),
                "fp32_one_minus_1e4": float(torch.tensor(1.) - 1e-4)}
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("expert_precision.json", {"expert_dtypes": summary, "unchanged_expert_tensors": unchanged}),
                            ("rounding_probe.json", rounding)):
        with (args.output / filename).open("x") as stream:
            json.dump(value, stream, indent=2)
    print(json.dumps({"expert_dtypes": summary, "rounding": rounding}, indent=2))


if __name__ == "__main__":
    main()