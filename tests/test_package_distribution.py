from __future__ import annotations

from email.parser import BytesParser
import os
from pathlib import Path
import site
import subprocess
import sys
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_built_wheel_has_only_public_dependencies_and_imports_without_private_sdk(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    environment = os.environ.copy()
    environment["PIP_NO_INDEX"] = "1"
    user_site = Path(site.getusersitepackages())
    if user_site.is_dir():
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(user_site), environment.get("PYTHONPATH", "")),
            )
        )
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(wheel_dir.glob("mp_comfyui_lab-*.whl"))

    with ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    requirements = metadata.get_all("Requires-Dist", [])

    assert requirements
    assert all("@" not in requirement for requirement in requirements)
    assert all("file:" not in requirement.casefold() for requirement in requirements)
    assert not any(
        "mp-runninghub-api" in requirement.casefold()
        for requirement in requirements
    )

    install_dir = tmp_path / "installed"
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(install_dir),
            str(wheel),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr

    script = f"""
import importlib.abc
import sys

class RejectPrivateSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'mp_runninghub_api' or fullname.startswith('mp_runninghub_api.'):
            raise AssertionError('private SDK import attempted')
        return None

sys.meta_path.insert(0, RejectPrivateSdk())
sys.path.insert(0, {str(install_dir)!r})
from mp_comfyui_lab.cli.app import main
from mp_comfyui_lab.executors.runninghub import RunningHubExecutor
assert RunningHubExecutor().name == 'runninghub'
assert int(main(['version'])) == 0
"""
    imported = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
