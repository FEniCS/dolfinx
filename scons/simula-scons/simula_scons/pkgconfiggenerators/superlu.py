#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

class unableToCompileException(Exception):
  def __init__(self, superlu_dir=""):
    Exception.__init__(self, "Unable to compile some superlu tests, using %s as location for superlu libs and includes" % (superlu_dir))

class unableToRunException(Exception):
  def __init__(self, superlu_dir="", arch=""):
    if arch == "darwin":
      Exception.__init__(self, "Unable to run superlu program. Maybe %s must be added in DYLD_LIBRARY_PATH" % (superlu_dir))
    else:
      Exception.__init__(self, "Unable to run superlu program. Maybe %s must be added in LD_LIBRARY_PATH or ld.so.conf" % (superlu_dir))

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
  """Run the tests for this package
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both
  """
  # Set architecture up front.
  arch = get_architecture()

  # Test program, check for superlu header.
  cpp_include_str = r"""
#include <dsp_defs.h>
#include <ssp_defs.h>
#include <csp_defs.h>
#include <zsp_defs.h>

int main() {
  return 0;
}
  
"""

  # Test program, check that I can call superlu functions
  cpp_testlib_str = r"""
#include <dsp_defs.h>

int main() {
  SuperMatrix A, L, U, B;
  SuperLUStat_t stat;
  int perm_c, perm_r;
  int stat, info;
  superlu_options_t options;
  set_default_options(&options);
  dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
}

"""

  write_cppfile(cpp_include_str, "superlu_config_test_include.cpp")
  write_cppfile(cpp_testlib_str, "superlu_config_test_libs.cpp")

  # Try to set superlu_dir from an evironment variable, else use some default.
  superlu_dir = ""
  if os.environ.has_key('SUPERLU_DIR'):
    superlu_dir = os.environ["SUPERLU_DIR"]
  else:
    superlu_dir = os.path.join(os.path.sep, "usr","local")

  # set which compiler and linker to use:
  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)
  
  cmdstr = "%s -I%s/include/superlu superlu_config_test_include.cpp" % (compiler, superlu_dir)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    raise unableToCompileException(superlu_dir)

  remove_cppfile("superlu_config_test_include.cpp", execfile=True)
  remove_cppfile("superlu_config_test_libs.cpp", execfile=True)


def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  pkgTests(sconsEnv=sconsEnv)

if __name__ == "__main__": 
  generatePkgConf(directory=".")
