#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getPetscDir(sconsEnv=None):
    petsc_dir = getPackageDir("petsc", sconsEnv=sconsEnv, default=None)
    if not petsc_dir:
        raise UnableToFindPackageException("PETSc")
    return petsc_dir

def getSlepcDir(sconsEnv=None):
    slepc_dir = getPackageDir("slepc", sconsEnv=sconsEnv, default=None)
    if not slepc_dir:
        raise UnableToFindPackageException("SLEPc")
    return slepc_dir

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
  """Run the tests for this package

     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker) 
     or just a string, which in that case will be used as both
  """
  arch = get_architecture()
  petsc_dir = getPetscDir(sconsEnv=sconsEnv)
  slepc_dir = getSlepcDir(sconsEnv=sconsEnv)

  # make sure that "directory" is contained in PKG_CONFIG_PATH, only relevant 
  # for test-cases where directory="."
  # FIXME: directory is not defined here
  if os.environ.has_key("PKG_CONFIG_PATH"):
    os.environ["PKG_CONFIG_PATH"] += ":%s" % os.getcwd()
  else:
    os.environ["PKG_CONFIG_PATH"] = "%s" % os.getcwd()

  # SLEPc depends on PETSC_DIR. 
  # prototype - make this a utility in commonPkgConfigUtils
  dep_module_name = "PETSc"
  dep_module = "petsc"
  notexist, cmdoutput = getstatusoutput("pkg-config --exists petsc")
  if notexist:
    # Try to generate petsc:
    ns = {}
    exec "import %s" % (dep_module) in ns
    packgen = ns.get("%s" % (dep_module))
    packgen.generatePkgConf(directory=os.getcwd(), sconsEnv=sconsEnv)
  # now the dep_module pkg-config should exist!
  failure,dep_mod_cflags = getstatusoutput("pkg-config %s --cflags" % (dep_module))
  if failure:
    # some strange unknown error, report something!
    raise UnableToXXXException("Unable to read CFLAGS for %s" % (dep_module_name))
  failure,dep_mod_libs = getstatusoutput("pkg-config %s --libs" % (dep_module))
  if failure:
    # some strange unknown error, report something!
    raise UnableToXXXException("Unable to read LDFLAGS for %s" % (dep_module_name))
  
  # Create a makefile to read basic things:
  slepc_makefile_str="""
# Retrive various flags from SLEPc settings.

PETSC_DIR=%s
SLEPC_DIR=%s

include ${SLEPC_DIR}/bmake/slepc_common

get_slepc_include:
	-@echo  ${SLEPC_INCLUDE}

get_slepc_libs:
	-@echo ${CC_LINKER_SLFLAG}${SLEPC_LIB_DIR} -L${SLEPC_LIB_DIR} -lslepc

get_petsc_arch:
	-@echo  ${PETSC_ARCH}
""" % (petsc_dir, slepc_dir)
  slepc_make_file = open("slepc_makefile","w")
  slepc_make_file.write(slepc_makefile_str)
  slepc_make_file.close()
  slepc_includes = ""
  slepc_libs = ""
  cmdstr = "make -s -f slepc_makefile get_slepc_include"
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    os.unlink("slepc_makefile")
    msg = "Unable to read SLEPc includes through make"
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  slepc_includes = cmdoutput

  cmdstr = "make -s -f slepc_makefile get_slepc_libs"
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    os.unlink("slepc_makefile")
    msg = "Unable to read SLEPc libs through make"
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  slepc_libs = cmdoutput

  cmdstr = "make -s -f slepc_makefile get_petsc_arch"
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    os.unlink("slepc_makefile")
    msg = "Unable to read SLEPc libs through make"
    raise UnableToXXXException(msg, errormsg=cmdoutput)
  petsc_arch = cmdoutput

  # Try to get compiler and linker from petsc
  cmdstr = "pkg-config petsc --variable=compiler"
  failure, cmdoutput = getstatusoutput(cmdstr)
  if failure:
    compiler = get_compiler(sconsEnv)
    print "Unable to get compiler from petsc.pc; using %s instead." % compiler
  else:
    compiler = cmdoutput
  cmdstr = "pkg-config petsc --variable=linker"
  failure, cmdoutput = getstatusoutput(cmdstr)
  if failure:
    linker = get_linker(sconsEnv)
    print "Unable to get linker from petsc.pc; using %s instead." % linker
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
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("slepc_config_test_version.cpp")
    raise UnableToCompileException("SLEPc", cmd=cmdstr,
                                   program=cpp_test_version_str, errormsg=cmdoutput)
  cmdstr = "%s %s %s slepc_config_test_version.o" % (linker, dep_mod_libs, slepc_libs)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile("slepc_config_test_version.cpp", ofile=True)
    raise UnableToLinkException("SLEPc", cmd=cmdstr,
                                program=cpp_test_version_str, errormsg=cmdoutput)
  if arch == 'darwin':
    os.putenv('DYLD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('DYLD_LIBRARY_PATH', ''),
                               os.path.join(slepc_dir, 'lib', petsc_arch)]))
  else:
    os.putenv('LD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('LD_LIBRARY_PATH', ''),
                               os.path.join(slepc_dir, 'lib', petsc_arch)]))
  runFailed, cmdoutput = getstatusoutput("./a.out")
  if runFailed:
    remove_cppfile("slepc_config_test_version.cpp", ofile=True, execfile=True)
    raise UnableToRunException("SLEPc", errormsg=cmdoutput)
  slepc_version = cmdoutput

  remove_cppfile("slepc_config_test_version.cpp", ofile=True, execfile=True)

  return slepc_version, slepc_libs, slepc_includes

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

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

