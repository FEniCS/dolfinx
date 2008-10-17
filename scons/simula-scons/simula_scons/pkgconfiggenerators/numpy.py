#!/usr/bin/env python

import os, sys
from os.path import sep, join, dirname, abspath

from commonPkgConfigUtils import *

# FIXME: script fails when running in the same folder as it is located

def trickPythonpath():
    # play tricks with pythonpath to avoid mixing up 
    # the real numpy with this file.
    file_abspath = abspath(dirname(__file__))
    try:
        splitpythonpath = os.environ['PYTHONPATH'].split(os.pathsep)
        index_filepath = splitpythonpath.index(file_abspath)
        del(splitpythonpath[index_filepath])
        os.environ['PYTHONPATH'] = os.path.pathsep.join(splitpythonpath)
    except:
        pass

def restorePythonpath():
    file_abspath = abspath(dirname(__file__))
    splitpythonpath = os.environ['PYTHONPATH'].split(os.pathsep)
    index_filepath = splitpythonpath.index(file_abspath)
    splitpythonpath.insert(index_filepath,file_abspath)
    os.environ['PYTHONPATH'] = os.path.pathsep.join(splitpythonpath)

def pkgVersion(**kwargs):
    #trickPythonpath()
    cmdstr = '%s -c "import numpy; print numpy.__version__"' % sys.executable
    failure, cmdoutput = getstatusoutput(cmdstr)
    if failure:
        msg = "Unable to get numpy version.\nCommand was:\n%s" % cmdstr
        raise UnableToXXXException(msg, errormsg=cmdoutput)
    else:
        version = cmdoutput
    #restorePythonpath()
    return version

def pkgCflags(**kwargs):
    #trickPythonpath()
    cmdstr = '%s -c "import numpy; print numpy.get_include()"' % sys.executable
    failure, cmdoutput = getstatusoutput(cmdstr)
    if failure:
        msg = "Unable to get numpy include folder.\nCommand was:\n%s" % cmdstr
        raise UnableToXXXException(msg, errormsg=cmdoutput)
    else:
        include_dir = cmdoutput
    #restorePythonpath()
    return "-I%s" % include_dir

def pkgLibs(**kwargs):
    return ""

def pkgTests(version=None, cflags=None, libs=None, **kwargs):

    if not cflags:
        cflags = pkgCflags()
    if not version:
        version = pkgVersion()
    if not libs:
        libs = pkgLibs()
        
    return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    # Generate a pkg-config file for numpy, put it in the given 
    # directory, if no directory is given, a suitable location is found 
    # using the functionality from commonPkgConfigUtils.

    version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

    major_version = version.split('.')[0]

    """Construct the content of the pkg-configfile"""
    pkg = """Name: NumPy
Description: Numerical Python
Version: %s
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
    # FIXME:    ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
    # Is there a better way to handle this on Windows?

    """Write the content to file, using correct version number"""
    file = open(os.path.join(directory,"numpy-%s.pc" % major_version), "w")
    file.write(pkg)
    file.close()
    print "done\n Found NumPy and generated pkg-config file in \n '%s'" % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
