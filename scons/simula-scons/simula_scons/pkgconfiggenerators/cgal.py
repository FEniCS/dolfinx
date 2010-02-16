#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getCgalDir(sconsEnv=None):
    return getPackageDir("cgal", sconsEnv=sconsEnv, default="/usr")

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
  # This is a bit special. It is given in the library as
  # a 10 digit number, like 1030511000. We have to do some arithmetics
  # to find the real version:
  # (VERSION / 1000 - 1000001) / 10000 => major version (3 in this case)
  # (VERSION / 1000 - 1000001) / 100 % 100 => minor version (5 in this case)
  # (VERSION / 1000 - 1000001) / 10 % 10 => sub-minor version (1 in this case).
  #
  # The version check also verify that we can include some CGAL headers.
    cpp_test_version_str = r"""
#include <CGAL/version.h>
#include <iostream>

int main() {
  #ifdef CGAL_VERSION_NR
    std::cout << CGAL_VERSION_NR;
  #endif
  return 0;
}
"""
    cppfile = "cgal_config_test_version.cpp"
    write_cppfile(cpp_test_version_str, cppfile);

    if not compiler:
        compiler = get_compiler(sconsEnv=sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)

    cmdstr = "%s -o a.out %s %s" % (compiler, cflags, cppfile)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cppfile)
        raise UnableToCompileException("CGAL", cmd=cmdstr,
                                       program=cpp_test_version_str,
                                       errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cppfile, execfile=True)
        raise UnableToRunException("CGAL", errormsg=cmdoutput)
    cgal_version_nr = int(cmdoutput)
    cgal_major = (cgal_version_nr / 1000 - 1000001) / 10000
    cgal_minor = (cgal_version_nr / 1000 - 1000001) / 100 % 100
    cgal_subminor = (cgal_version_nr / 1000 - 1000001) / 10 % 10
    full_cgal_version = "%s.%s.%s" % (cgal_major, cgal_minor, cgal_subminor)

    remove_cppfile(cppfile, execfile=True)

    return full_cgal_version

def pkgCflags(sconsEnv=None):
    return "-I%s -frounding-math" % \
           os.path.join(getCgalDir(sconsEnv=sconsEnv), "include")

def pkgLibs(sconsEnv=None):
    return "-L%s -lCGAL -lCGAL_Core" % \
           os.path.join(getCgalDir(sconsEnv), "lib")

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
    """Run the tests for this package

    If Ok, return various variables, if not we will end with an exception.
    forceCompiler, if set, should be a tuple containing (compiler, linker)
    or just a string, which in that case will be used as both
    """

    if not forceCompiler:
        compiler = get_compiler(sconsEnv)
        linker = get_linker(sconsEnv)
    else:
        compiler, linker = set_forced_compiler(forceCompiler)

    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)
    if not libs:
        libs = pkgLibs(sconsEnv=sconsEnv)
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv, compiler=compiler,
                             linker=linker, cflags=cflags, libs=libs)
    else:
        # FIXME: Add a real test for this package
        pkgVersion(sconsEnv=sconsEnv, compiler=compiler,
                   linker=linker, cflags=cflags, libs=libs)

    return version, libs, cflags

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):
    if directory is None:
        directory = suitablePkgConfDir()

    version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: cgal
Version: %s
Description: Computational Geometry Algorithms Library
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
    # FIXME:      ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
    # Is there a better way to handle this on Windows?

    pkg_file = open(os.path.join(directory, "cgal.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found cgal and generated pkg-config file in\n '%s'" \
          % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
