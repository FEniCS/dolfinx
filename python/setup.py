import os
import shlex
import subprocess
import sys
import sysconfig

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if sys.version_info < (3, 8):
    print("Python 3.8 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.7.0.dev0"

REQUIREMENTS = [
    "cffi",
    "numpy>=1.21",
    "mpi4py",
    "petsc4py",
    "fenics-ffcx>=0.7.0.dev0,<0.8.0",
    "fenics-ufl>=2023.2.0.dev0,<2023.3.0"
]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


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
        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                       '-DPython3_EXECUTABLE=' + sys.executable,
                       f'-DPython3_INCLUDE_DIRS={sysconfig.get_config_var("INCLUDEPY")}']
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        env = os.environ.copy()
        env['pybind11_DIR'] = pybind11.get_cmake_dir()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cmake_generator = env.get("CMAKE_GENERATOR", "")
        cmake_args += ["-B", self.build_temp, "-S", ext.sourcedir]
        if not cmake_generator:
            try:
                # Use ninja if available
                s = subprocess.run(['cmake', '-G Ninja'] + cmake_args, capture_output=True, check=True, env=env)
                sys.stderr.write(s.stderr)
                sys.stdout.write(s.stdout)
            except (FileNotFoundError, subprocess.CalledProcessError):
                if "CMAKE_BUILD_PARALLEL_LEVEL" not in env:
                    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "3"
                subprocess.run(['cmake'] + cmake_args, env=env)
        else:
            subprocess.run(['cmake'] + cmake_args, env=env)
        subprocess.run(['cmake', '--build', self.build_temp], env=env)


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
      setup_requires=["pybind11"],
      zip_safe=False)
