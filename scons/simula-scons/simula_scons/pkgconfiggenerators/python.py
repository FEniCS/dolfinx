#!/usr/bin/env python
import os
import distutils 
import sys

from os.path import sep, join
from distutils import sysconfig



from commonPkgConfigUtils import *

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

    """Construct the content of the pkg-configfile"""
    pkg = """Name: %s
Description: %s
Version: %s
Requires: %s
Libs: %s
Cflags: %s
""" % (name , desc, full_version, reqs, libs, cflags)

    """Write the content to file, using correct version number"""
    file = open("%s/python-%s.pc" % (directory,str(major_version)[0]), "w")
    file.write(pkg)
    file.close()
    
    print "done\n Found '%s' and generated pkg-config file in \n %s" % ("Python", directory)

if __name__ == "__main__":
    generatePkgConf(directory=".")
