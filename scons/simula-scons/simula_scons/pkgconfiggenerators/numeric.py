#!/usr/bin/env python
import Numeric
import os
from os.path import sep, join

from commonPkgConfigUtils import *

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
    return True

# Generate a pkg-config file for numeric, put it in the given 
# directory, if no directory is given, a suitable location is found 
# using the functionality from commonPkgConfigUtils.
# Nothing in kwargs is used, it is included to ensure a consitent 
# interface for the generatePkgConf function.
def generatePkgConf(directory=suitablePkgConfDir(), **kwargs):
    """Use Numeric.__file__ to find out where Python is installed, and set
    variables."""

    print "Generating pkg-config file for '%s' in %s" % ("numeric", directory)

    fullpath = Numeric.__file__
    #pref, lib, py = fullpath.split(os.path.sep)[1:4]
    splitfullpath = fullpath.split(sep)
    # it is safe to assume that 'site-packages' is somewhere in there...
    sitepackind = splitfullpath.index('site-packages')
    # the python-version is the one just infront of site-packages
    py = splitfullpath[sitepackind-1]
    # lib is before that
    lib = splitfullpath[sitepackind-2]
    # prefix is from the beginning up to lib
    pref = splitfullpath[1:sitepackind-2]
    prefix      = sep + sep.join(pref)
    libdir      = sep + lib
    includedir  = sep + "include"
    pydir       = sep + py + sep

    """Set some relevant pkg-config file values. Are all these required?"""
    name = "Numeric"
    desc = "Numerical Python"
    version = Numeric.__version__
    reqs = ""
    libs = ""
    cflags = "-I${includedir}%s" % pydir 

    """Construct the content of the pkg-configfile"""
    pkg = """prefix=%s
exec_prefix=${prefix}
libdir=${exec_prefix}%s
includedir=${prefix}%s

Name: %s
Description: %s
Version: %s
Requires: %s
Libs: %s
Cflags: %s
""" % (prefix, libdir, includedir, name , desc, version, reqs, libs, cflags)

    """Write the content to file, using correct version number"""
    file = open("%s/numeric-%s.pc" % (directory,str(version)[:2]), "w")
    file.write(pkg)
    file.close()


if __name__ == "__main__":
    generatePkgConf(directory=".")
