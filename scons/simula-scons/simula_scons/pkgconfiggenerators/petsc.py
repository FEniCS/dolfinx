#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

def getPetscVariables(variables=('includes','libs','compiler','linker'), sconsEnv=None):
  if isinstance(variables, str):
    variables = (variables,)
  filename = "petsc_makefile"
  arch = get_architecture()
  # Create a makefile to read basic things from PETSc
  petsc_makefile_str="""
# Retrive various flags from PETSc settings.

PETSC_DIR=%s

include ${PETSC_DIR}/bmake/common/variables

get_petsc_include:
	-@echo  -I${PETSC_DIR}/bmake/${PETSC_ARCH} -I${PETSC_DIR}/include ${MPI_INCLUDE}

get_petsc_libs:
	-@echo   ${C_SH_LIB_PATH} -L${PETSC_LIB_DIR} ${PETSC_LIB_BASIC}

get_petsc_cc:
	-@echo ${PCC}

get_petsc_ld:
	-@echo ${PCC_LINKER}
""" % getPetscDir(sconsEnv=sconsEnv)
  petsc_make_file = open(filename, "w")
  petsc_make_file.write(petsc_makefile_str)
  petsc_make_file.close()

  petsc_includes = None
  if 'includes' in variables: 
    cmdstr = "make -s -f %s get_petsc_include" % filename
    runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if runFailed:
      os.unlink(filename)
      msg = "Unable to read PETSc includes through make."
      raise UnableToXXXException(msg, errormsg=cmdoutput)
    petsc_includes = cmdoutput

  petsc_libs = None
  if 'libs' in variables:
    cmdstr = "make -s -f %s get_petsc_libs" % filename
    runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if runFailed:
      os.unlink(filename)
      msg = "Unable to read PETSc libs through make."
      raise UnableToXXXException(msg, errormsg=cmdoutput)
    petsc_libs = cmdoutput

  petsc_cc = None
  if 'compiler' in variables:
    cmdstr = "make -s -f %s get_petsc_cc" % filename
    runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if runFailed:
      os.unlink(filename)
      msg = "Unable to figure out correct PETSc compiler."
      raise UnableToXXXException(msg, errormsg=cmdoutput)
    # We probably get a c-compiler from the petsc config, switch to a 
    # compatible c++
    petsc_cc = cmdoutput
    if not is_cxx_compiler(petsc_cc):
      petsc_cc = get_alternate_cxx_compiler(petsc_cc, arch=arch)

  petsc_ld = None
  if 'linker' in variables:
    cmdstr = "make -s -f %s get_petsc_ld" % filename
    runFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if runFailed:
      os.unlink(filename)
      msg = "Unable to figure out correct PETSc linker"
      raise UnableToXXXException(msg, errormsg=cmdoutput)
    # We probably get a c-compiler from the petsc config, switch to a 
    # compatible c++
    petsc_ld = cmdoutput
    if not is_cxx_compiler(petsc_ld):
      petsc_ld = get_alternate_cxx_compiler(petsc_ld, arch=arch)

  os.unlink(filename)

  ret = []
  for variable in petsc_includes, petsc_libs, petsc_cc, petsc_ld:
    if variable is not None:
      ret.append(variable)
  return ret

def getPetscDir(sconsEnv=None):
    petsc_dir = getPackageDir("petsc", sconsEnv=sconsEnv, default=None)
    if not petsc_dir:
        petsc_locations = ["/usr/lib/petscdir/2.3.3"]
        for petsc_location in petsc_locations:
            if os.access(petsc_location, os.F_OK) == True:
                return petsc_location
        raise UnableToFindPackageException("PETSc")
    return petsc_dir

def pkgVersion(compiler=None, linker=None, cflags=None, libs=None, sconsEnv=None):
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
    compiler = pkgCompiler(sconsEnv=sconsEnv)
  if not linker:
    linker = pkgLinker(sconsEnv=sconsEnv)
  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv=sconsEnv)

  cmdstr = "%s %s -c petsc_config_test_version.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("petsc_config_test_version.cpp")
    raise UnableToCompileException("PETSc", cmd=cmdstr,
                                   program=cpp_test_version_str, errormsg=cmdoutput)
  cmdstr = "%s %s petsc_config_test_version.o" % (linker, libs)
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile("petsc_config_test_version.cpp", ofile=True)
    raise UnableToLinkException("PETSc", cmd=cmdstr,
                                program=cpp_test_version_str, errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cppfile("petsc_config_test_version.cpp", ofile=True, execfile=True)
    raise UnableToRunException("PETSc", errormsg=cmdoutput)
  petsc_version = cmdoutput

  remove_cppfile("petsc_config_test_version.cpp", ofile=True, execfile=True)
  return petsc_version

def pkgLibs(sconsEnv=None):
  libs, = getPetscVariables(variables='libs', sconsEnv=sconsEnv)
  return libs

def pkgCflags(sconsEnv=None):
  includes, = getPetscVariables(variables='includes', sconsEnv=sconsEnv)
  return includes

def pkgCompiler(sconsEnv=None):
  compiler, = getPetscVariables(variables='compiler', sconsEnv=sconsEnv)
  if not compiler:
    compiler = get_compiler(sconsEnv)
  return compiler

def pkgLinker(sconsEnv=None):
  linker, = getPetscVariables(variables='linker', sconsEnv=sconsEnv)
  if not linker:
    linker = get_linker(sconsEnv)
  return linker

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
  """Run the tests for this package.
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both.
  """
  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv=sconsEnv)

  if not forceCompiler:
    compiler = pkgCompiler(sconsEnv=sconsEnv)
    linker = pkgLinker(sconsEnv=sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  if not version:
    version = pkgVersion(cflags=cflags, libs=libs,
                         compiler=compiler, linker=linker, sconsEnv=sconsEnv)
  else:
    # Run pkgVersion since this is the current PETSc test
    pkgVersion(cflags=cflags, libs=libs,
               compiler=compiler, linker=linker, sconsEnv=sconsEnv)

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
