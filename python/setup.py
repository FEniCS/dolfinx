import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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

        # TODO: Not sure about the logic of this. cmake needs to be able to
        # just find C++ pybind11 if its a non-Python install.
        if "pybind11_DIR" not in os.environ:
            try:
                import pybind11
                cmake_args += [f'-DPYBIND11_DIR={pybind11.get_cmake_dir()}']
            except ImportError:
                pass

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Default to 3 build threads
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ['-j3']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(ext_modules=[CMakeExtension('dolfinx.cpp')],
      cmdclass=dict(build_ext=CMakeBuild))
