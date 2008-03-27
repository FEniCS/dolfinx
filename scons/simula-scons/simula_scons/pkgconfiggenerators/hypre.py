#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

def getHypreDir():
  # Try to get hypre_dir from an evironment variable, else use some default.
  if os.environ.has_key('HYPRE_DIR'):
    hypre_dir = os.environ["HYPRE_DIR"]
  else:
    # FIXME: should /usr be the default Hypre dir (as in the Debian package)?
    hypre_dir = os.path.join(os.path.sep,"usr","local")
  return hypre_dir

def pkgVersion(compiler=None, cflags=None, **kwargs):
  # A test for checking the hypre version
  cpp_version_str = r"""
#include <stdio.h>
#include <HYPRE.h>
#include <HYPRE_config.h>

int main() {
  #ifdef HYPRE_PACKAGE_VERSION
    printf("%s",HYPRE_PACKAGE_VERSION);
  #endif
  return 0;
}
"""
  write_cppfile(cpp_version_str, "hypre_config_test_version.c")

  if not compiler:
    # FIXME: get_compiler() returns a C++ compiler. Should be a C compiler.
    compiler = get_compiler(kwargs.get("sconsEnv", None))
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s %s hypre_config_test_version.c" % (compiler,cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_version.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_version_str, errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cfile("hypre_config_test_version.c", execfile=True)
    raise UnableToRunException("Hypre", errormsg=cmdoutput)
  hypre_version = cmdoutput

  remove_cfile("hypre_config_test_version.c", execfile=True)

  return hypre_version

def pkgLibs(compiler=None, cflags=None, **kwargs):
  # A test to see if we need to link with blas and lapack
  cpp_libs_str = r"""
#include <stdio.h>
#include <HYPRE.h>
#include <HYPRE_config.h>

int main() {
  #ifdef HAVE_BLAS
    printf("-lblas\n");
  #endif
  #ifdef HAVE_LAPACK
    printf("-llapack\n");
  #endif
  return 0;
}
"""
  write_cppfile(cpp_libs_str, "hypre_config_test_libs.c")

  if not compiler:
    # FIXME: get_compiler() returns a C++ compiler. Should be a C compiler.
    compiler = get_compiler(kwargs.get("sconsEnv", None))
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s %s hypre_config_test_libs.c" % (compiler,cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_libs.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_libs_str, errormsg=cmdoutput)
  
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cfile("hypre_config_test_libs.c", execfile=True)
    raise UnableToRunException("Hypre", errormsg=cmdoutput)
  libs_str = string.join(string.split(cmdoutput, '\n'))
  if 'blas' in libs_str and 'lapack' in libs_str:
    # Add only -llapack since lapack should be linked to the rigth blas lib
    # FIXME: Is this always true?
    libs_str = '-llapack'
    
  remove_cfile("hypre_config_test_libs.c", execfile=True)

  libs_dir = os.path.join(getHypreDir(),"lib")
  libs_str += " -L%s -lHYPRE" % libs_dir
  
  # Set ATLAS location from environ, or use default
  atlaslocation = ""
  if os.environ.has_key('ATLAS_DIR'):
    atlaslocation = os.environ["ATLAS_DIR"]
  else:
    atlaslocation = os.path.join(os.path.sep, "usr","lib","atlas")

  if get_architecture() == "darwin":
    libs_str += " -framework vecLib"
  else:
    libs_str += " -L%s" % atlaslocation

  return libs_str

def pkgCflags(**kwargs):
  include_dir = os.path.join(getHypreDir(),"include")
  cflags = "-I%s -fPIC" % include_dir
  if get_architecture() == "darwin":
    # Additional cflags required on Mac
    cflags += " -fno_common"
  return cflags

def pkgTests(forceCompiler=None, sconsEnv=None,
             version=None, libs=None, cflags=None, **kwargs):
  """Run the tests for this package.
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both
  """
  # Set architecture up front.
  arch = get_architecture()

  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)
  # FIXME: should we have, e.g., a get_c_compiler that returns gcc as default?
  # using gcc as compiler and linker for now
  compiler = 'gcc'
  linker = 'gcc'

  if not cflags:
    cflags = pkgCflags()
  if not version:
    version = pkgVersion(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)

  # A test to see if we can actually use the libHYPRE
  cpp_testlib_str = r"""
#include <stdio.h>
#include <HYPRE.h>
#include <HYPRE_config.h>
#include <_hypre_IJ_mv.h>

int main() {
  MPI_Comm comm = MPI_COMM_NULL;
  HYPRE_IJMatrix A;
  HYPRE_IJMatrixCreate(comm, 0, 10, 0, 10, &A);
  return 0;
}
"""
  write_cppfile(cpp_testlib_str, "hypre_config_test_libstest.c")

  cmdstr = "%s %s -c hypre_config_test_libstest.c" % (compiler,cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_libstest.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_testlib_str, errormsg=cmdoutput)
  
  cmdstr = "%s %s hypre_config_test_libstest.o" % (linker,libs)
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    remove_cfile("hypre_config_test_libstest.c", ofile=True)
    raise UnableToLinkException("Hypre", cmd=cmdstr,
                                program=cpp_testlib_str, errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cfile("hypre_config_test_libstest.c", execfile=True, ofile=True)
    raise UnableToRunException("Hypre", errormsg=cmdoutput)
  
  remove_cfile("hypre_config_test_libstest.c", execfile=True, ofile=True)

  return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
  """Generate a pkg-config file for Hypre."""
  
  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: Hypre
Version: %s
Description: The Hypre project - http://www.llnl.gov/casc/hypre/software.html
Libs: %s
Cflags: %s
""" % (version, libs, cflags)
  
  pkg_file = open(os.path.join(directory,"hypre.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found Hypre and generated pkg-config file in \n '%s'" % directory

if __name__ == "__main__":
  generatePkgConf(directory=".")
