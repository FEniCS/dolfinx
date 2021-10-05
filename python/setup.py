import os
import platform
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if sys.version_info < (3, 7):
    print("Python 3.7 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "0.3.1.dev0"

REQUIREMENTS = [
    "numpy",
    "mpi4py",
    "petsc4py",
    "fenics-ffcx>=0.3.1.dev0,<0.4.0",
    "fenics-ufl>=2021.1.0,<2021.2.0"
]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPython3_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j3']

        env = os.environ.copy()
        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


setup(name='fenics-dolfinx',
      version=VERSION,
      author='FEniCS Project',
      description='DOLFINx Python interface',
      long_description='',
      packages=["dolfinx",
                "dolfinx.fem",
                "dolfinx.wrappers",
                "dolfinx_utils.test"],
      package_data={'dolfinx.wrappers': ['*.h']},
      ext_modules=[CMakeExtension('dolfinx.cpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      setup_requires=["pybind11"],
      zip_safe=False)
