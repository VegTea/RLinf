"""Static interface checks for IsaacLab benchmark launchers."""

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPO_ROOT / "examples/embodiment/check_isaaclab_gpu_smoke.sh",
    REPO_ROOT / "examples/embodiment/benchmark_isaaclab_h100_scaling.sh",
    REPO_ROOT / "examples/embodiment/submit_isaaclab_h100_benchmark_qzcli.sh",
)


def test_isaaclab_benchmark_scripts_have_valid_bash_syntax() -> None:
    """Benchmark launchers should fail fast on shell syntax errors."""
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_h100_benchmark_help_describes_two_gpu_usage() -> None:
    """The benchmark launcher exposes the intended single-node GPU controls."""
    script = REPO_ROOT / "examples/embodiment/benchmark_isaaclab_h100_scaling.sh"
    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--gpus N" in result.stdout
    assert "--selected-env N" in result.stdout
