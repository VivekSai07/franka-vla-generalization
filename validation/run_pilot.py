"""Run the approved pilot, then validate; never train, upload, resume or retry a failed job."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.benchmark import write_json
from data_collection.pilot_plan import validate_plan, plan_digest
from validation.gates import require_replay_pass
from validation.validate_pilot import validate_bank
from src.robots.franka_panda.config import SCENE_PATH
import mujoco


def main():
    directory = ROOT / "outputs/pilot_overnight_20260915"
    directory.mkdir(exist_ok=False)
    dataset = ROOT / "datasets/franka_pilot_20260915"
    plans = ROOT / "outputs/benchmark_preparation_limits_20260915"
    plan_path = plans / "pilot_plan.json"
    bank = plans / "pilot_scenes"
    report = {"status": "preflight", "dataset": str(dataset), "plan": str(plan_path),
              "scene_bank": str(bank), "expected_train": 300, "expected_validation": 120}
    write_json(directory / "run.json", report)
    try:
        require_replay_pass(ROOT / "datasets/franka_pilot_preparation_limits_20260915", ROOT)
        plan = json.loads(plan_path.read_text())
        validate_plan(plan)
        if plan["mode"] != "pilot" or len(plan["slots"]) != 420 or dataset.exists():
            raise ValueError("Incorrect pilot plan or output already exists")
        if shutil.disk_usage(ROOT).free < 30 * 1024**3:
            raise ValueError("Less than 30 GiB available; refusing overnight launch")
        report["plan_digest"] = plan_digest(plan)
        report["bank_counts"] = validate_bank(bank, plan, mujoco.MjModel.from_xml_path(SCENE_PATH), set())
        report["status"] = "collecting"
        write_json(directory / "run.json", report)
        commands = [
            [sys.executable, "data_collection/collect_libero_dataset.py", "--headless",
             "--plan", str(plan_path), "--root", str(dataset)],
            [sys.executable, "validation/validate_pilot.py", "--root", str(dataset), "--scene-bank", str(bank)],
        ]
        for phase, command in zip(("collecting", "validating"), commands):
            report["status"] = phase
            write_json(directory / "run.json", report)
            print(f"PILOT PHASE: {phase}", flush=True)
            result = subprocess.run(command, cwd=ROOT)
            report[f"{phase}_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{phase} failed with exit {result.returncode}; no automatic retry")
        final = json.loads((dataset / "validation_result.json").read_text())
        require_replay_pass(dataset, ROOT)
        if final["episode_count"] != 420 or final["split_counts"] != {"train": 300, "validation": 120}:
            raise ValueError("Unexpected final count/split")
        report.update(status="passed", episode_count=final["episode_count"],
                      frame_count=final["frame_count"], split_counts=final["split_counts"])
        write_json(directory / "run.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        write_json(directory / "run.json", report)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()