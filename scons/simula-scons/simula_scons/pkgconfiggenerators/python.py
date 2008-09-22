#!/usr/bin/env python
import os
import distutils 
import sys

from os.path import sep, join
from distutils import sysconfig

from commonPkgConfigUtils import *

# The following methods (get_pythonlib_dir and get_pythonlib_name)
# are borrowed from NumPy (http://projects.scipy.org/scipy/numpy/browser/branches/numpy.scons/numpy/distutils/scons/core/extension_scons.py)
# and modified somewhat:

def get_pythonlib_dir():
    """Return the path to look for the python engine library
    (pythonXY.lib on win32, libpythonX.Y.so on unix, etc...)."""
    if os.name == 'nt':
        return os.path.join(sys.exec_prefix, 'libs')
    else:
        return os.path.join(sys.exec_prefix, 'lib')

def get_pythonlib_name(debug = 0):
    """Return the name of Python library (necessary to link on
    NT with MinGW)."""
    return "python%d%d" % (sys.hexversion >> 24,
                           (sys.hexversion >> 16) & 0xff)

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
    return True

# Generate a pkg-config file for python, put it in the given 
# directory, if no directory is given, a suitable location is found 
# using the functionality from commonPkgConfigUtils.
# Nothing in kwargs is used, it is included to ensure a consitent 
# interface for the generatePkgConf function.
def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

    name = "Python"
    desc = "The Python library"
    #version = distutils.__version__
    full_version = ".".join([str(s) for s in sys.version_info[0:3]])
    major_version = sys.version_info[0]
    reqs = ""
    libs = ""
    cflags = "-I%s" % sysconfig.get_python_inc() 

    # The Python library is required on the Windows platform:
    if sys.platform.startswith("win"):
        libs += "-L%s -l%s" % (get_pythonlib_dir(),
                               get_pythonlib_name())

    """Construct the content of the pkg-configfile"""
    pkg = """Name: %s
Description: %s
Version: %s
Requires: %s
Libs: %s
Cflags: %s
""" % (name , desc, full_version, reqs, repr(libs)[1:-1], repr(cflags)[1:-1])
    # FIXME: is there a better way?     ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^

    """Write the content to file, using correct version number"""
    file = open(join(directory,"python-%s.pc" % str(major_version)[0]), "w")
    file.write(pkg)
    file.close()
    
    print "done\n Found '%s' and generated pkg-config file in \n %s" % ("Python", directory)

if __name__ == "__main__":
    generatePkgConf(directory=".")
