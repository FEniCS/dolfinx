#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

class unableToFindPackage(Exception):
  def __init__(self, package="", what=""):
    Exception.__init__(self, "Unable to find the location of %s (%s), consider setting the SLEPC_DIR variable" % (package, what))

class unableToCompileException(Exception):
  def __init__(self, slepc_dir="", errormsg=""):
    Exception.__init__(self, "Unable to compile some slepc tests, using %s as location for slepc includes.\nAdditional message:\n%s" % (slepc_dir, errormsg))
    
class unableToLinkException(Exception):
  def __init__(self, slepc_dir="", errormsg=""):
    Exception.__init__(self, "Unable to link some slepc tests, using %s as location for slepc libs.\nAdditional message:\n%s" % (slepc_dir, errormsg))

class unableToRunException(Exception):
  def __init__(self):
    Exception.__init__(self, "Unable to run slepc program correctly.")


def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
  """Run the tests for this package

     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker) 
     or just a string, which in that case will be used as both
  """
  arch = get_architecture()

  # make sure that "directory" is contained in PKG_CONFIG_PATH, only relevant 
  # for test-cases where directory="."
  # FIXME: directory is not defined here
##   if os.environ.has_key("PKG_CONFIG_PATH"):
##     os.environ["PKG_CONFIG_PATH"] += ":%s" % (directory)
##   else:
##     os.environ["PKG_CONFIG_PATH"] = "%s" % (directory)

  # SLEPc depends on PETSC_DIR. 
  # prototype - make this a utility in commonPkgConfigUtils
  dep_module_name = "PETSc"
  dep_module = "petsc"
  notexist, cmdoutput = commands.getstatusoutput("pkg-config --exists petsc")
  if notexist:
    # Try to generate petsc:
    # Are we running "globally"?
    try:
      packgen = __import__("simula_scons.pkgconfiggenerators", globals(), locals())
    except:
      print "[scons/pkgconfiggenerators/slepc.py. l. 40] Unable to find pkgconfiggenerators"
      raise Exception ("Unable to find the pkgconfiggenerators")
    ns = {}
    exec "from simula_scons.pkgconfiggenerators import %s" % (dep_module) in ns
    packgen = ns.get("%s" % (dep_module))
    #packgen.generatePkgConf(directory=directory, sconsEnv=sconsEnv)
    packgen.generatePkgConf(sconsEnv=sconsEnv)
  # now the dep_module pkg-config should exist!
  failure,dep_mod_cflags = commands.getstatusoutput("pkg-config %s --cflags" % (dep_module))
  if failure:
    # some strange unknown error, report something!
    raise Exception ("Unable to read CFLAGS for %s" % (dep_module_name))
  failure,dep_mod_libs = commands.getstatusoutput("pkg-config %s --libs" % (dep_module))
  if failure:
    # some strange unknown error, report something!
    raise Exception ("Unable to read LDFLAGS for %s" % (dep_module_name))

  slepc_dir = ""
  if os.environ.has_key("SLEPC_DIR"):
    slepc_dir = os.environ["SLEPC_DIR"]
  else:
    raise Exception ("The SLEPC_DIR environment variable is not set. Plese set the variable to point to your SLEPc installation")
 
  
  # Create a makefile to read basic things:
  slepc_makefile_str="""
# Retrive various flags from SLEPc settings.

include ${SLEPC_DIR}/bmake/slepc_common

get_slepc_include:
	-@echo  ${SLEPC_INCLUDE}

get_slepc_libs:
	-@echo ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${C_SH_LIB_PATH} -L${PETSC_LIB_DIR} ${PETSC_LIB_BASIC}
"""
  slepc_make_file = open("slepc_makefile","w")
  slepc_make_file.write(slepc_makefile_str)
  slepc_make_file.close()
  slepc_includes = ""
  slepc_libs = ""
  runFailed, cmdoutput = commands.getstatusoutput("make -f slepc_makefile get_slepc_include")
  if runFailed:
    raise Exception ("Unable to read SLEPc includes through make")
  slepc_includes = cmdoutput
    
  runFailed, cmdoutput = commands.getstatusoutput("make -f slepc_makefile get_slepc_libs")
  if runFailed:
    raise Exception ("Unable to read SLEPc libs through make")
  slepc_libs = cmdoutput

  # Try to get compiler and linker from petsc
  failure, cmdoutput = commands.getstatusoutput("pkg-config petsc --variable=compiler")
  if failure:
    compiler = get_compiler(sconsEnv)
    print "Could not retrieve compiler info from petsc.pc; using %s instead." % compiler
  else:
    compiler = cmdoutput
  failure, cmdoutput = commands.getstatusoutput("pkg-config petsc --variable=linker")
  if failure:
    linker = get_linker(sconsEnv)
    print "Could not retrieve linker info from petsc.pc; using %s instead." % linker
  else:
    linker = cmdoutput

  os.unlink("slepc_makefile")

  cpp_test_version_str = r"""
#include <stdio.h>
#include <slepc.h>

int main() {
  #ifdef SLEPC_VERSION_MAJOR
    #ifdef SLEPC_VERSION_MINOR
      #ifdef SLEPC_VERSION_SUBMINOR
        printf("%d.%d.%d", SLEPC_VERSION_MAJOR, SLEPC_VERSION_MINOR, SLEPC_VERSION_SUBMINOR);
      #else
        printf("%d.%d", SLEPC_VERSION_MAJOR, SLEPC_VERSION_MINOR);
      #endif
    #else
      printf("%d", SLEPC_VERSION_MAJOR);
    #endif
  #endif
  return 0;
}
"""

  write_cppfile(cpp_test_version_str, "slepc_config_test_version.cpp");

  cmdstr = "%s %s %s -c slepc_config_test_version.cpp" % (compiler, dep_mod_cflags, slepc_includes)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    raise unableToCompileException(slepc_dir=slepc_dir, errormsg=cmdoutput)
  cmdstr = "%s %s slepc_config_test_version.o" % (linker, slepc_libs)
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    raise unableToLinkException(slepc_dir=slepc_dir, errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    raise unableToRunException()
  slepc_version = cmdoutput

  remove_cppfile("slepc_config_test_version.cpp", ofile=True, execfile=True)

  return slepc_version, slepc_libs, slepc_includes

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  (slepc_version, slepc_libs, slepc_includes) = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: SLEPc
Version: %s
Description: The SLEPc project from Universidad Politecnica de Valencia, Spain
Requires: petsc
Libs: %s
Cflags: %s
""" % (slepc_version, slepc_libs, slepc_includes)
  pkg_file = open("%s/slepc.pc" % (directory), "w")
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found '%s' and generated pkg-config file in \n %s" % ("SLEPc", directory)

if __name__ == "__main__": 
  generatePkgConf(directory=".")

