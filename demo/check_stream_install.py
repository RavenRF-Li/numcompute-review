"""Build and smoke-test the project wheel without modifying the aiml environment."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv


ROOT = Path(__file__).resolve().parents[1]
SMOKE = r"""
from importlib.metadata import requires
from pathlib import Path
import re
import sys

import numcompute
import numcompute_stream
from numcompute_stream import DecisionTreeClassifier, Pipeline, StreamTrainer
from numcompute_stream.visualise import plot_metric_over_time

for package in (numcompute, numcompute_stream):
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), package.__file__
    print(package.__file__)
requirements = requires("numcompute")
assert {re.split(r"[<>=!~;\s]", item, maxsplit=1)[0] for item in requirements} == {"numpy", "matplotlib"}
print("Wheel requirements:", requirements)

trainer = StreamTrainer(Pipeline([("tree", DecisionTreeClassifier(buffer_size=2))]), labels=[0, 1])
assert trainer.fit_chunk([[0], [1]], [0, 1])["status"] == "warmup"
scored = trainer.score_chunk([[0], [1]], [0, 1])
assert scored["y_pred"].tolist() == [0, 1] and scored["metrics"]["accuracy"] == 1
out = Path("installed-smoke.png")
plot_metric_over_time([1], "Installed wheel smoke test", "Accuracy", save_path=out)
assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
print("Installed training, prediction and Agg PNG checks passed.")
"""


def main():
    with TemporaryDirectory(prefix="numcompute-wheel-") as directory:
        scratch = Path(directory)
        source = scratch / "source"
        source.mkdir()
        shutil.copy2(ROOT / "pyproject.toml", source)
        for package in ("numcompute", "numcompute_stream"):
            shutil.copytree(ROOT / package, source / package, ignore=shutil.ignore_patterns("__pycache__"))

        env = {
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(scratch / "matplotlib"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
        subprocess.run(
            [sys.executable, "-B", "-m", "pip", "wheel", "--no-index", "--no-deps",
             "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(scratch), str(source)],
            cwd=scratch, env=env, check=True,
        )
        wheel, = scratch.glob("numcompute-*.whl")
        environment = scratch / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-B", "-m", "pip", "install", "--no-index", "--no-deps",
             "--ignore-installed", "--no-cache-dir", str(wheel)],
            cwd=scratch, env=env, check=True,
        )
        subprocess.run([str(python), "-IB", "-c", SMOKE], cwd=scratch, env=env, check=True)


if __name__ == "__main__":
    main()
