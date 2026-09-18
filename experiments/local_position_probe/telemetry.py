"""Local, devicewide/shared-GPU measurements; never reads tensors or touches RNG."""
import csv
from datetime import datetime, timezone
import json
import math
import subprocess
import threading
import time

QUERY = ['nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
         '--format=csv,noheader,nounits']
FIELDS = ('utilization_percent', 'memory_used_mib', 'memory_total_mib', 'power_w', 'temperature_c')


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse(text):
    rows = []
    for values in csv.reader(text.splitlines()):
        if len(values) != 7: raise ValueError('Unexpected nvidia-smi column count')
        row = dict(uuid=values[0].strip(), name=values[1].strip(), missing={})
        if not row['uuid'] or not row['name']: raise ValueError('GPU identity missing')
        for key, raw in zip(FIELDS, values[2:]):
            try:
                number = float(raw)
                if not math.isfinite(number): raise ValueError('nonfinite')
                row[key] = number
            except ValueError:
                row[key] = None
                row['missing'][key] = 'unavailable_or_nonfinite'
        rows.append(row)
    if not rows: raise ValueError('No NVML device rows')
    return rows


def query():
    result = subprocess.run(QUERY, capture_output=True, text=True, check=True, timeout=3)
    return parse(result.stdout)


class Sampler:
    def __init__(self, path, phase):
        self.path, self.phase = path, phase
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.errors = []
        self.thread = threading.Thread(target=self._loop, name='local-position-telemetry', daemon=True)

    def sample(self):
        row = dict(utc=utc(), monotonic_seconds=time.monotonic(), phase=self.phase(),
                   scope='devicewide_shared_gpu', interval_seconds=5, devices=[], error=None)
        try: row['devices'] = query()
        except Exception as exc:
            # Type only: exception text may contain command/environment details.
            row['error'] = type(exc).__name__
        with self.lock:
            if row['error']: self.errors.append(dict(utc=row['utc'], error=row['error']))
            try:
                with self.path.open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
            except OSError as exc:
                self.errors.append(dict(utc=row['utc'], error=type(exc).__name__, stage='telemetry_write'))

    def _loop(self):
        deadline = time.monotonic()
        while not self.stop.is_set():
            self.sample()
            deadline += 5
            if self.stop.wait(max(0., deadline-time.monotonic())): break

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()  # Query timeout bounds cleanup; never leaves a live sampler.