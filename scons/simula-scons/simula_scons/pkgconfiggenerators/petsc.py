#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

class UnableToFindPackageException(UnableToXXXException):
  def __init__(self, package=""):
    msg = ("Unable to find the location of %s. Consider setting " + \
           "the %s_DIR variable.") % (package, package.upper())
    UnableToXXXException.__init__(self, msg)

class UnableToCompileException(UnableToXXXException):
  def __init__(self, petsc_dir="", errormsg=""):
    msg = ("Unable to compile some PETSc tests, using '%s' as " + \
           "location for PETSc includes.") % petsc_dir
    UnableToXXXException.__init__(self, msg, errormsg=errormsg)
    
class UnableToLinkException(UnableToXXXException):
  def __init__(self, petsc_dir="", errormsg=""):
    msg = ("Unable to link some PETSc tests, using '%s' as location " + \
           "for PETSc libs.") % petsc_dir
    UnableToXXXException.__init__(self, msg, errormsg=errormsg)

class UnableToRunException(UnableToXXXException):
  def __init__(self, package, errormsg=""):
    msg = "Unable to run %s program correctly." % package
    UnableToXXXException.__init__(self, msg, errormsg=errormsg)

def getPetscVariables():
  # First make sure PETSC_DIR is set. This will raise an exception if not:
  petsc_dir = getPetscDir()
  filename = "petsc_makefile"
  arch = get_architecture()
  # Create a makefile to read basic things from PETSc
  petsc_makefile_str="""
# Retrive various flags from PETSc settings.

include ${PETSC_DIR}/bmake/common/variables

get_petsc_include:
	-@echo  -I${PETSC_DIR}/bmake/${PETSC_ARCH} -I${PETSC_DIR}/include ${MPI_INCLUDE}

get_petsc_libs:
	-@echo   ${C_SH_LIB_PATH} -L${PETSC_LIB_DIR} ${PETSC_LIB_BASIC}

get_petsc_cc:
	-@echo ${PCC}

get_petsc_ld:
	-@echo ${PCC_LINKER}
"""
  petsc_make_file = open(filename, "w")
  petsc_make_file.write(petsc_makefile_str)
  petsc_make_file.close()

  cmdstr = "make -f %s get_petsc_include" % filename
  runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if runFailed:
    msg = "Unable to read PETSc includes through make."
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  petsc_includes = cmdoutput

  cmdstr = "make -f %s get_petsc_libs" % filename
  runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if runFailed:
    msg = "Unable to read PETSc libs through make."
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  petsc_libs = cmdoutput

  cmdstr = "make -f %s get_petsc_cc" % filename
  runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if runFailed:
    msg = "Unable to figure out correct PETSc compiler."
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  # We probably get a c-compiler from the petsc config, switch to a 
  # compatible c++
  petsc_cc = get_alternate_cxx_compiler(cmdoutput,arch=arch)

  cmdstr = "make -f %s get_petsc_ld" % filename
  runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if runFailed:
    msg = "Unable to figure out correct PETSc linker"
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  # We probably get a c-compiler from the petsc config, switch to a 
  # compatible c++
  petsc_ld = get_alternate_cxx_compiler(cmdoutput,arch=arch)

  os.unlink(filename)
  
  return petsc_includes, petsc_libs, petsc_cc, petsc_ld

def getPetscDir():
  petsc_dir = ""
  if os.environ.has_key("PETSC_DIR"):
    petsc_dir = os.environ["PETSC_DIR"]
  else:
    #petsc_dir = os.path.join(os.path.sep,"usr","lib","petsc")
    raise UnableToFindPackageException(package="PETSc")
  return petsc_dir 

def pkgVersion(compiler=None, linker=None, cflags=None, libs=None, **kwargs):
  cpp_test_version_str = r"""
#include <stdio.h>
#include <petsc.h>

int main() {
  #ifdef PETSC_VERSION_MAJOR
    #ifdef PETSC_VERSION_MINOR
      #ifdef PETSC_VERSION_SUBMINOR
        printf("%d.%d.%d", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR);
      #else
        printf("%d.%d", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR);
      #endif
    #else
      printf("%d", PETSC_VERSION_MAJOR);
    #endif
  #endif
  return 0;
}
"""
  write_cppfile(cpp_test_version_str, "petsc_config_test_version.cpp");

  if not compiler:
    compiler = pkgCompiler(**kwargs)
  if not linker:
    linker = pkgLinker(**kwargs)
  if not cflags:
    cflags = pkgCflags()
  if not libs:
    libs = pkgLibs()

  cmdstr = "%s %s -c petsc_config_test_version.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    raise UnableToCompileException(petsc_dir=getPetscDir(), errormsg=cmdoutput)
  #cmdstr = "%s %s petsc_config_test_version.o" % (linker, libs)
  cmdstr = "%s petsc_config_test_version.o" % linker
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    raise UnableToLinkException(petsc_dir=getPetscDir(), errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    raise UnableToRunException("PETSc", errormsg=cmdoutput)
  petsc_version = cmdoutput

  remove_cppfile("petsc_config_test_version.cpp", ofile=True, execfile=True)
  return petsc_version

def pkgLibs(**kwargs):
  includes, libs, compiler, linker = getPetscVariables()
  libs = " ".join(set(libs.split()))  # remove duplicates
  return libs

def pkgCflags(**kwargs):
  includes, libs, compiler, linker = getPetscVariables()
  cflags = " ".join(set(includes.split()))  # remove duplicates
  return cflags

def pkgCompiler(**kwargs):
  includes, libs, compiler, linker = getPetscVariables()
  if not compiler:
    compiler = get_compiler(kwargs.get("sconsEnv", None))
  return compiler

def pkgLinker(**kwargs):
  includes, libs, compiler, linker = getPetscVariables()
  if not linker:
    linker = get_linker(kwargs.get("sconsEnv", None))
  return linker

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
  """Run the tests for this package.
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both.
  """
  if not cflags:
    cflags = pkgCflags()
  if not libs:
    libs = pkgLibs()

  if not forceCompiler:
    compiler = pkgCompiler(sconsEnv=sconsEnv)
    linker = pkgLinker(sconsEnv=sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  if not version:
    version = pkgVersion(cflags=cflags, libs=libs,
                         compiler=compiler, linker=linker)
  else:
    # Run pkgVersion since this is the current PETSc test
    pkgVersion(cflags=cflags, libs=libs, compiler=compiler, linker=linker)

  return version, compiler, linker, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  version, compiler, linker, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: PETSc
Version: %s
Description: The PETSc project from Argonne Nat.Lab, Math. and CS Division (http://www-unix.mcs.anl.gov/petsc/petsc-as/)
compiler=%s
linker=%s
Libs: %s
Cflags: %s
""" % (version, compiler, linker, libs, cflags)
  pkg_file = open(os.path.join(directory,"petsc.pc"), "w")
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  
  print "done\n Found PETSc and generated pkg-config file in\n '%s'" % directory

if __name__ == "__main__": 
  generatePkgConf(directory=".")
