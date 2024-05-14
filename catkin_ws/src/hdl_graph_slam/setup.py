## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
# pylint: disable=missing-module-docstring
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["hdl_graph_slam"], package_dir={"": "src"}
)

setup(**setup_args)
