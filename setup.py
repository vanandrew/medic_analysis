import subprocess
from pathlib import Path
from setuptools import setup

THISDIR = Path(__file__).parent

# get scripts path
scripts_path = THISDIR / "medic_analysis" / "scripts"

setup(
    entry_points={
        "console_scripts": [
            f"{f.stem}=medic_analysis.scripts.{f.stem}:main"
            for f in scripts_path.glob("*.py")
            if f.name not in "__init__.py"
        ]
    },
)
