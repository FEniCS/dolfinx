#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getZlibDir(sconsEnv=None):
    return getPackageDir("zlib", sconsEnv=sconsEnv, default="/usr")

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
    cpp_test_version_str = r"""
#include <stdio.h>
#include <zlib.h>

int main() {
  #ifdef ZLIB_VER_MAJOR
    #ifdef ZLIB_VER_MINOR
      #ifdef ZLIB_VER_REVISION
        printf("%d.%d.%d", ZLIB_VER_MAJOR, ZLIB_VER_MINOR, ZLIB_VER_REVISION);
      #else
        printf("%d.%d", ZLIB_VER_MAJOR, ZLIB_VER_MINOR);
      #endif
    #else
      printf("%d", ZLIB_VER_MAJOR);
    #endif
  #endif
  return 0;
}
"""
    cppfile = "zlib_config_test_version.cpp"
    write_cppfile(cpp_test_version_str, cppfile);

    if not compiler:
        compiler = get_compiler(sconsEnv=sconsEnv)
    if not linker:
        compiler = get_linker(sconsEnv=sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)
    if not libs:
        libs = pkgLibs(sconsEnv=sconsEnv)

    cmdstr = "%s %s -c %s" % (compiler, cflags, cppfile)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cppfile)
        raise UnableToCompileException("zlib", cmd=cmdstr,
                                       program=cpp_test_version_str,
                                       errormsg=cmdoutput)

    cmdstr = "%s -o a.out %s %s" % (linker, libs, cppfile.replace('.cpp', '.o'))
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        remove_cppfile(cppfile, ofile=True)
        raise UnableToLinkException("zlib", cmd=cmdstr,
                                    program=cpp_test_version_str,
                                    errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cppfile, ofile=True, execfile=True)
        raise UnableToRunException("zlib", errormsg=cmdoutput)
    version = cmdoutput

    remove_cppfile(cppfile, ofile=True, execfile=True)
    return version

def pkgCflags(sconsEnv=None):
    return "-I%s" % os.path.join(getZlibDir(sconsEnv=sconsEnv), "include")

def pkgLibs(sconsEnv=None):
    return "-L%s -lz" % os.path.join(getZlibDir(sconsEnv), "lib")

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

    pkg_file_str = r"""Name: zlib
Version: %s
Description: data-compression library
Libs: %s
Cflags: %s
""" % (version, libs, cflags)

    pkg_file = open(os.path.join(directory, "zlib.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found zlib and generated pkg-config file in\n '%s'" \
          % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
