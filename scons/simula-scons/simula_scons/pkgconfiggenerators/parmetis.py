#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

def getParmetisDir(sconsEnv=None):
    parmetis_dir = getPackageDir("parmetis", sconsEnv=sconsEnv,
                                 default=os.path.join(os.path.sep, "usr"))
    return parmetis_dir

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
  cpp_test_version_str = r"""
#include <stdio.h>
#include <parmetis.h>

int main() {
  #ifdef PARMETIS_MAJOR_VERSION
    #ifdef PARMETIS_MINOR_VERSION
      printf("%d.%d", PARMETIS_MAJOR_VERSION, PARMETIS_MINOR_VERSION);
    #else
      printf("%d", PARMETIS_MAJOR_VERSION);
    #endif
  #endif
  return 0;
}
"""
  cppfile = "parmetis_config_test_version.cpp"
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
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile(cppfile)
    raise UnableToCompileException("ParMETIS", cmd=cmdstr,
                                   program=cpp_test_version_str,
                                   errormsg=cmdoutput)
  cmdstr = "%s %s %s" % (linker, libs, cppfile.replace('.cpp', '.o'))
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile(cppfile, ofile=True)
    raise UnableToLinkException("ParMETIS", cmd=cmdstr,
                                program=cpp_test_version_str,
                                errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cppfile(cppfile, ofile=True, execfile=True)
    raise UnableToRunException("ParMETIS", errormsg=cmdoutput)
  version = cmdoutput

  remove_cppfile(cppfile, ofile=True, execfile=True)
  return version

def pkgCflags(sconsEnv=None):
    return "-I%s" % os.path.join(getParmetisDir(sconsEnv=sconsEnv), "include")

def pkgLibs(sconsEnv=None):
  return "-L%s -lparmetis -lmetis" % \
         os.path.join(getParmetisDir(sconsEnv), "lib")

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

    # FIXME: Add test for this package

    return version, cflags, libs

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    
    version, cflags, libs = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: ParMETIS
Version: %s
Description: Parallel Graph Partitioning and Fill-reducing Matrix Ordering
Libs: %s
Cflags: %s
""" % (version, libs, cflags)

    pkg_file = open(os.path.join(directory, "parmetis.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found ParMETIS and generated pkg-config file in\n '%s'" % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
