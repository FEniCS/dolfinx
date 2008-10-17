#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getGmpDir(sconsEnv=None):
    gmp_dir = getPackageDir("gmp", sconsEnv=sconsEnv,
                            default=os.path.join(os.path.sep, "usr"))
    return gmp_dir

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
  cpp_test_version_str = r"""
#include <stdio.h>
#include <gmpxx.h>

int main() {
  #ifdef __GNU_MP_VERSION
    #ifdef __GNU_MP_VERSION_MINOR
      #ifdef __GNU_MP_VERSION_PATCHLEVEL
        printf("%d.%d.%d", __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR, __GNU_MP_VERSION_PATCHLEVEL);
      #else
        printf("%d.%d", __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR);
      #endif
    #else
      printf("%d", __GNU_MP_VERSION);
    #endif
  #endif
  return 0;
}
"""
  cppfile = "gmp_config_test_version.cpp"
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
    raise UnableToCompileException("GMP", cmd=cmdstr,
                                   program=cpp_test_version_str,
                                   errormsg=cmdoutput)

  cmdstr = "%s -o a.out %s %s" % (linker, libs, cppfile.replace('.cpp', '.o'))
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile(cppfile, ofile=True)
    raise UnableToLinkException("GMP", cmd=cmdstr,
                                program=cpp_test_version_str,
                                errormsg=cmdoutput)

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cppfile(cppfile, ofile=True, execfile=True)
    raise UnableToRunException("GMP", errormsg=cmdoutput)
  version = cmdoutput

  remove_cppfile(cppfile, ofile=True, execfile=True)
  return version

def pkgCflags(sconsEnv=None):
    return "-I%s" % os.path.join(getGmpDir(sconsEnv=sconsEnv), "include")

def pkgLibs(sconsEnv=None):
  return "-L%s -lgmp -lgmpxx" % os.path.join(getGmpDir(sconsEnv), "lib")

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

    # A program that do a real GMP test (thanks to Benjamin Kehlet)
    cpp_test_lib_str = r"""
#include <iostream>
#include <gmpxx.h>

int main (void)
{
  mpz_class integer1("1000000023457323");
  mpz_class integer2("54367543212");
  mpz_class int_result = integer1*integer2;

  std::cout << integer1 << " * " << integer2 << " = "
            << int_result << std::endl;

  return EXIT_SUCCESS;
}
"""
    cpp_file = "gmp_config_test_lib.cpp"
    write_cppfile(cpp_test_lib_str, cpp_file);

    # try to compile the simple GMP test
    cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("GMP", cmd=cmdstr,
                                       program=cpp_test_lib_str,
                                       errormsg=cmdoutput)

    cmdstr = "%s -o a.out %s %s" % \
             (linker, libs, cpp_file.replace('.cpp', '.o'))
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        remove_cppfile(cpp_file, ofile=True)
        raise UnableToLinkException("GMP", cmd=cmdstr,
                                    program=cpp_test_lib_str,
                                    errormsg=errormsg)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cpp_file, ofile=True, execfile=True)
        raise UnableToRunException("GMP", errormsg=cmdoutput)
    try:
        value = cmdoutput.split("=")[1].strip()
        assert value == '54367544487317021840341476'
    except:
        errormsg = "GMP test does not produce correct result, " \
                   "check your GMP installation."
        errormsg += "\n%s" % cmdoutput
        raise UnableToRunException("GMP", errormsg=errormsg)

    remove_cppfile(cpp_file, ofile=True, execfile=True)

    return version, cflags, libs

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    
    version, cflags, libs = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: GNU MP
Version: %s
Description: GNU Multiple Precision Arithmetic Library
Libs: %s
Cflags: %s
""" % (version, libs, cflags)

    pkg_file = open(os.path.join(directory, "gmp.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found GMP and generated pkg-config file in\n '%s'" % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
