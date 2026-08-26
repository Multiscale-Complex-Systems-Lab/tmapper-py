"""One (N, path) measurement, in its own process.

Run per-process because psutil's peak_wset is a high-water mark for the
whole process and never decreases: measuring dense and blocked in one
process makes every run after the first report the first one's peak.

usage: measure_one.py <N> <dense|blocked>
prints: N path wall_s peak_gb nodes edges
"""
import importlib.util
import sys
import time
from pathlib import Path

import psutil

REPO = Path(r"C:\Dropbox\MSU\code\toolboxes\tmapper-py")
sys.path.insert(0, str(REPO / "src"))

from tmapper import sample_data_path  # noqa: E402

n = int(sys.argv[1])
path = sys.argv[2]

spec = importlib.util.spec_from_file_location(
    "tm_app", REPO / "src" / "tmapper" / "app" / "streamlit_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

df, _ = app.read_csv_smart(sample_data_path())
app.LOW_MEMORY_ROWS = 0 if path == "blocked" else 10**9

proc = psutil.Process()
base = proc.memory_info().peak_wset
t0 = time.perf_counter()
built = app.build_network(
    df, ("tmax", "tmin", "prcp"), True, len(df) - n, len(df) - 1, 1, 0, 1,
    3, 3.0, 30, 100.0, 0.5, True,
)
wall = time.perf_counter() - t0
peak = (proc.memory_info().peak_wset - base) / 1e9
g = built["g_simp"]
print(f"{n} {path} {wall:.3f} {peak:.3f} "
      f"{g.number_of_nodes()} {g.number_of_edges()}")
