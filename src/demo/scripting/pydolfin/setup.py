#!/usr/bin/env python

# Ideas from Berthold Höllmann:
# http://mail.python.org/pipermail/distutils-sig/2000-October/001658.html

from distutils.core import setup, Extension
from distutils.command import install_data

class my_install_data(install_data.install_data):
    def finalize_options (self):
        self.set_undefined_options('install',
                                   ('install_lib', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                   )

setup(name="PyDOLFIN",
      version="0.0.1",
      description="DOLFIN Python module",
      author="Johan Jansson (Python wrapper)",
      author_email="dolfin@fenics.org",
      url="http://www.fenics.org/dolfin/",
      packages=["dolfin/"],
      scripts=[],
      data_files=[('dolfin', ["dolfin/_dolfin.so"])],
      cmdclass = {'install_data': my_install_data}
      )
