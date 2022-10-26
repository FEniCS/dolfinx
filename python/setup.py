import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION = "0.5.2.dev0"
REQUIREMENTS = [
    "cffi",
    "numpy>=1.21",
    "mpi4py",
    "petsc4py",
    "fenics-ffcx>=0.5.1.dev0,<0.6.0",
    "fenics-ufl>=2022.3.0.dev0,<2022.4.0"
]


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
                       f'-DPython3_EXECUTABLE={sys.executable}',
                       f'-DPython3_INCLUDE_DIRS={sysconfig.get_config_var("INCLUDEPY")}']

        # if "pybind11_DIR" not in os.environ:
        import pybind11
        cmake_args += [f'-DPYBIND11_DIR={pybind11.get_cmake_dir()}']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Default to 3 build threads
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ['-j3']

        # import pybind11
        # env = os.environ.copy()
        # env['pybind11_DIR'] = pybind11.get_cmake_dir()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(name='fenics-dolfinx',
      version=VERSION,
      author='FEniCS Project',
      description='DOLFINx Python interface',
      long_description='',
      packages=["dolfinx",
                "dolfinx.fem",
                "dolfinx.io",
                "dolfinx.nls",
                "dolfinx.wrappers"],
      package_data={'dolfinx.wrappers': ['*.h'], 'dolfinx': ['py.typed'],
                    'dolfinx.fem': ['py.typed'], 'dolfinx.nls': ['py.typed']},
      ext_modules=[CMakeExtension('dolfinx.cpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      zip_safe=False,
      python_requires=">=3.8",)
