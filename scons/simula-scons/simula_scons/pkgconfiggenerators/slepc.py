#!/usr/bin/env python
import os,sys
import string
import os.path

import petsc

from commonPkgConfigUtils import *

def getSlepcIncAndLibs(sconsEnv=None):
  slepc_dir = getSlepcDir(sconsEnv=sconsEnv)

  if os.path.exists(os.path.join(slepc_dir, "bmake")):
      slepc_path_variables = "bmake"
  elif os.path.exists(os.path.join(slepc_dir, "conf")):
      slepc_path_variables = "conf"
  else:
      raise UnableToFindPackageException("SLEPc")

  # Create a makefile to read basic things:
  slepc_makefile_str="""
# Retrive various flags from SLEPc settings.

PETSC_DIR=%s
PETSC_ARCH=%s
SLEPC_DIR=%s

include ${SLEPC_DIR}/%s/slepc_common

get_slepc_include:
	-@echo  ${SLEPC_INCLUDE}

get_slepc_libs:
	-@echo ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc ${SLEPC_EXTERNAL_LIB}
""" % (petsc.getPetscDir(sconsEnv=sconsEnv),
       petsc.getPetscArch(sconsEnv=sconsEnv),
       slepc_dir, slepc_path_variables)
  slepc_make_file = open("slepc_makefile", "w")
  slepc_make_file.write(slepc_makefile_str)
  slepc_make_file.close()

  cmdstr = "make -s -f slepc_makefile get_slepc_include"
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    # Unable to read SLEPc config through make; trying defaults.
    slepc_includes = "-I%s" % os.path.join(slepc_dir, 'include', 'slepc')
  else:
    slepc_includes = cmdoutput

  if not runFailed:
    cmdstr = "make -s -f slepc_makefile get_slepc_libs"
    runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    slepc_libs = "-lslepc"
  else:
    slepc_libs = cmdoutput

  os.unlink("slepc_makefile")

  return slepc_includes, slepc_libs

def getSlepcDir(sconsEnv=None):
    slepc_dir = getPackageDir("slepc", sconsEnv=sconsEnv, default=None)
    if not slepc_dir:
        slepc_locations = ["/usr/lib/slepcdir/3.0.0", "/usr/lib/slepcdir/2.3.3"]
        for slepc_location in slepc_locations:
            if os.access(slepc_location, os.F_OK) == True:
                return slepc_location
        raise UnableToFindPackageException("SLEPc")
    return slepc_dir

def pkgVersion(compiler=None, linker=None, cflags=None, libs=None, sconsEnv=None):
  """Return SLEPc version"""
  if not compiler:
    compiler = pkgCompiler(sconsEnv=sconsEnv)
  if not linker:
    linker = pkgLinker(sconsEnv=sconsEnv)
  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv=sconsEnv)

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

  cmdstr = "%s %s %s -c slepc_config_test_version.cpp" % \
           (compiler, petsc.pkgCflags(sconsEnv=sconsEnv), cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("slepc_config_test_version.cpp")
    raise UnableToCompileException("SLEPc", cmd=cmdstr,
                                   program=cpp_test_version_str,
                                   errormsg=cmdoutput)

  cmdstr = "%s -o a.out %s %s slepc_config_test_version.o" % \
           (linker, petsc.pkgLibs(sconsEnv=sconsEnv), libs)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile("slepc_config_test_version.cpp", ofile=True)
    raise UnableToLinkException("SLEPc", cmd=cmdstr,
                                program=cpp_test_version_str, errormsg=cmdoutput)

  slepc_dir = getSlepcDir(sconsEnv=sconsEnv)
  petsc_arch = petsc.getPetscArch(sconsEnv=sconsEnv)
  if get_architecture() == 'darwin':
    os.putenv('DYLD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('DYLD_LIBRARY_PATH', ''),
                               os.path.join(slepc_dir, 'lib', petsc_arch)]))
  else:
    os.putenv('LD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('LD_LIBRARY_PATH', ''),
                               os.path.join(slepc_dir, 'lib', petsc_arch)]))

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cppfile("slepc_config_test_version.cpp", ofile=True, execfile=True)
    raise UnableToRunException("SLEPc", errormsg=cmdoutput)
  slepc_version = cmdoutput

  remove_cppfile("slepc_config_test_version.cpp", ofile=True, execfile=True)

  return slepc_version

def pkgCflags(sconsEnv=None):
    return getSlepcIncAndLibs(sconsEnv=sconsEnv)[0]

def pkgLibs(sconsEnv=None):
    return getSlepcIncAndLibs(sconsEnv=sconsEnv)[1]

def pkgCompiler(sconsEnv=None):
  return petsc.pkgCompiler(sconsEnv=sconsEnv)

def pkgLinker(sconsEnv=None):
  return petsc.pkgLinker(sconsEnv=sconsEnv)

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
    # Run pkgVersion since this is the current SLEPc test
    pkgVersion(cflags=cflags, libs=libs,
               compiler=compiler, linker=linker, sconsEnv=sconsEnv)

  return version, libs, cflags

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):

  if directory is None:
    directory = suitablePkgConfDir()

  slepc_version, slepc_libs, slepc_includes = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: SLEPc
Version: %s
Description: The SLEPc project from Universidad Politecnica de Valencia, Spain
Requires: petsc
Libs: %s
Cflags: %s
""" % (slepc_version, slepc_libs, slepc_includes)
  pkg_file = open("%s/slepc.pc" % directory, "w")
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found SLEPc and generated pkg-config file in \n '%s'" % directory

if __name__ == "__main__": 
  generatePkgConf(directory=".")

