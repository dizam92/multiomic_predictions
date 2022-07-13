
import glob
import os
import sys
from distutils.core import setup

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "python")))

setup(
    name="MultiOmicTransformer",
    description="MOT",
    author="Mazid OSSENI",
    version="0.1.0",
    packages=["multiomic_modeling"],
    scripts=glob.glob("bin/*"),
    license="For research usage only",
)
