#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getMtl4Dir(sconsEnv=None):
  arch = get_architecture()
  if arch == "darwin":
    default = '/sw'  # use fink as default
  elif arch.startswith('win'):
    default = 'C:'
  else:
    default = '/usr'
  mtl4_dir = getPackageDir("mtl4", sconsEnv=sconsEnv, default=default)
  return mtl4_dir

def pkgVersion(compiler=None, cflags=None, sconsEnv=None):
  """Find the MTL4 version."""
  # FIXME: Maybe we should use MTL4_DIR/VERSION.INPUT to figure out the version?
  return '4'

def pkgLibs(sconsEnv=None):
  return ""

def pkgCflags(sconsEnv=None):
  include_dir = None
  mtl4_dir = getMtl4Dir(sconsEnv=sconsEnv)
  for inc_dir in "", "include":
    if os.path.isfile(os.path.join(mtl4_dir, inc_dir,
                                   "boost", "numeric", "mtl", "mtl.hpp")):
      include_dir = inc_dir
      break
  if include_dir is None:
    raise UnableToFindPackageException("MTL4")
  return "-I%s" % os.path.join(mtl4_dir, include_dir)

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
  """Run the tests for this package.
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both.
  """
  # set which compiler and linker to use:
  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
  if not version:
    version = pkgVersion(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv=sconsEnv)
  
  # All we want to do is to compile in some MTL4 headers, so really know
  # enough already, as the API of the headers are defined by the version.
  cpp_test_str = r"""
#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
int main() {
  mtl::dense_vector<double> x(10);
  if (x.size() == 10) {
    std::cout << "mtl4 ok";
  } else {
    std::cout << "mtl4 not ok";
  }
}
"""
  write_cppfile(cpp_test_str, "mtl4_config_test.cpp")

  cmdstr = "%s -o a.out %s mtl4_config_test.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("mtl4_config_test.cpp")
    raise UnableToCompileException("MTL4", cmd=cmdstr,
                                   program=cpp_test_str, errormsg=cmdoutput)

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed or cmdoutput != "mtl4 ok":
    remove_cppfile("mtl4_config_test.cpp", execfile=True)
    raise UnableToRunException("MTL4", errormsg=cmdoutput)

  remove_cppfile("mtl4_config_test.cpp", execfile=True)

  return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  name = "MTL4"
  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: %s
Version: %s
Description: Matrix Template Library
Libs: %s
Cflags: %s
""" % (name, version, libs, repr(cflags)[1:-1])
  # FIXME:                  ^^^^^^^^^^^^^^^^^^
  # Is there a better way to handle this on Windows?
  
  pkg_file = open(os.path.join(directory, "mtl4.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found %s and generated pkg-config file in\n '%s'" % \
        (name, directory)

if __name__ == "__main__":
  generatePkgConf(directory=".")
