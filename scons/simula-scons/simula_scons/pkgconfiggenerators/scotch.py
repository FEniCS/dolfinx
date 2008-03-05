#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

class unableToCompileException(Exception):
  def __init__(self, scotch_dir="", cmdstr=None):
    if cmdstr:
      Exception.__init__(self, "Unable to compile a scotch tests,\nusing %s as location for scotch includes.\nTry to set the SCOTCH_DIR environment variable.\nThe following compile command was used:\n%s" % (scotch_dir, cmdstr))
    else:
      Exception.__init__(self, "Unable to compile a scotch tests,\nusing %s as location for scotch includes.\nTry to set the SCOTCH_DIR environment variable." % (scotch_dir))

class unableToLinkException(Exception):
  def __init__(self, scotch_dir="", cmdstr=None):
    if cmdstr:
      Exception.__init__(self, "Unable to link a scotch test,\nusing %s as location for scotch includes.\nTry to set the SCOTCH_DIR environment variable.\nThe following link command was used:\n%s" % (scotch_dir, cmdstr))
    else:
      Exception.__init__(self, "Unable to link a scotch tests,\nusing %s as location for scotch includes.\nTry to set the SCOTCH_DIR environment variable." % (scotch_dir))


class unableToRunException(Exception):
  def __init__(self):
    Exception.__init__(self, "Unable to run scotch program correctly.")

class runFailureException(Exception):
  def __init__(self):
    Exception.__init__(self, "The scotch testprogram does not run correctly.")

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
  """Run the tests for this package
  
  If Ok, return various variables, if not we will end with an exception.
  forceCompiler, if set, should be a tuple containing (compiler, linker)
  or just a string, which in that case will be used as both
  """

  # There are two different ways of installing scotch, as done by scotch just
  # in one directory, or in subdirs lib, include, etc...
  # If usealternate is True, the latter is assumed. Default is the former.
  usealternate = False

  # Don't know of a way to figure out the scotch version, have to investigate 
  # that later. From what I can tell, scotch itself does not support this.
  # We will just assume scotch v.4.0

  # It seems that the version number is present in scotch.h (and ptscotch.h).
  # Maybe we can use that?
  
  # Simple test-program that try to include the main scotch header
  cpp_test_include_str = r"""
extern "C" {
#include <common.h>
#include <scotch.h>
}
int main() {
  return 0;
}
"""

  # Test that I can call a scotch function
  cpp_test_lib_str = r"""
extern "C" {
#include <common.h>
#include <scotch.h>
}
#include <iostream>
int main() {
  SCOTCH_Mesh mesh;

  if (SCOTCH_meshInit (&mesh) !=0) {
    std::cout<<"failure";
  } else {
    std::cout<<"success";
  }
  return 0;
}
"""

  write_cppfile(cpp_test_include_str,"scotch_config_test_include.cpp")
  write_cppfile(cpp_test_lib_str, "scotch_config_test_lib.cpp")
 
  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  scotch_dir=""
  if os.environ.has_key('SCOTCH_DIR'):
    scotch_dir = os.environ['SCOTCH_DIR']
  else:
    scotch_dir = os.path.join(os.path.sep, "usr", "local", "scotch")
  scotch_libs = scotch_dir
  scotch_includes = scotch_dir

  # test that we can include scotch:
  cmdstr = "%s -I%s -c scotch_config_test_include.cpp" % (compiler, scotch_dir)
  alt_cmdstr = "%s -I%s/include -c scotch_config_test_include.cpp" % (compiler, scotch_dir)
  # first we try to look in the traditional subdirs lib and include
  compileFailed, cmdoutput = commands.getstatusoutput(alt_cmdstr)
  if compileFailed:
    # then we look under scotch_dir
    compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if compileFailed:
      # Nope, still does not work, exit with Exception, print orig. cmdstr
      raise unableToCompileException(scotch_dir, cmdstr)
  else:
    usealternate = True
    scotch_libs = os.path.join(scotch_libs,'lib')
    scotch_includes = os.path.join(scotch_includes,'include')

  remove_cppfile("scotch_config_test_include.cpp", ofile=True)

  # test that we can compile a program using scotch:
  cmdstr = "%s -I%s -c scotch_config_test_lib.cpp" % (compiler, scotch_dir)
  alt_cmdstr = "%s -I%s/include -c scotch_config_test_lib.cpp" % (compiler, scotch_dir)
  if usealternate:
    cmdstr = alt_cmdstr
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    raise unableToCompileException(scotch_dir, cmdstr)

  # test that we can link a binary using scotch:
  cmdstr = "%s -L%s scotch_config_test_lib.o -lscotch -lscotcherrcom -lcommon" % (linker, scotch_dir)
  alt_cmdstr = "%s -L%s/lib scotch_config_test_lib.o -lscotch -lscotcherrcom -lcommon" % (linker, scotch_dir)
  if usealternate:
    cmdstr = alt_cmdstr
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    raise unableToLinkException(scotch_dir, cmdstr)
   
  # run the binary:
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    raise unableToRunException()
  if not cmdoutput == "success":
    raise runFailureException()

  remove_cppfile("scotch_config_test_lib.cpp", ofile=True, execfile=True)

  # Trying to figure out the version number from scotch.h
  if usealternate:
    scotch_h_file = open("%s/include/scotch.h" % scotch_dir, 'r')
  else:
    scotch_h_file = open("%s/scotch.h" % scotch_dir, 'r')
  scotch_version = '4.0'  # assume 4.0 as default
  for line in scotch_h_file:
    if "Version" in line:
      # the lines with the version number is something like this:
      # /**                # Version 4.0  : from : 11 dec 2001     **/
      tmp = line.split()
      scotch_version = tmp[tmp.index("Version")+1]

  return scotch_version, scotch_libs, scotch_includes

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  scotch_version, scotch_libs, scotch_includes = pkgTests(sconsEnv=sconsEnv)

  # Ready to create a pkg-config file:
  pkg_file_str = r"""Name: scotch
Version: %s
Description: Scotch mesh and graph partitioning, http://www.labri.fr/perso/pelegrin/scotch/
Libs: -L%s -lscotch -lscotcherrcom -lcommon
Cflags: -I%s 
""" % (scotch_version, scotch_libs, scotch_includes)
  pkg_file = open("%s/scotch.pc" % (directory), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found '%s' and generated pkg-config file in\n '%s'" % ("scotch", directory)

if __name__ == "__main__":
  generatePkgConf(directory=".")
