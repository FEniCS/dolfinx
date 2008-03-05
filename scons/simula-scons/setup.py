# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils import sysconfig
import sys, os, glob

setup(name='simula-scons',
    version='0.1',
    description='Addons for building software with SCons',
    author='Åsmund Ødegård, Arve N. Knudsen, Ola Skavhaug',
    author_email='aasmund@simula.no',
    url='http://www.simula.no',
    packages=['simula_scons', 'simula_scons.pkgconfiggenerators'],
    package_dir={'simula_scons': 'simula_scons', 'simula_scons.pkgconfiggenerators': 'simula_scons/pkgconfiggenerators'},
    )
