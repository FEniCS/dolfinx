#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getHypreDir(sconsEnv=None):
    default = os.path.join(os.path.sep,"usr","local")
    hypre_dir = getPackageDir("hypre", sconsEnv=sconsEnv, default=default)
    return hypre_dir

def pkgVersion(compiler=None, linker=None, cflags=None, sconsEnv=None):
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
    compiler = get_compiler(sconsEnv)
  if not linker:
    linker = get_linker(sconsEnv)
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s -o a.out %s hypre_config_test_version.c" % (compiler, cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_version.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_version_str, errormsg=cmdoutput)
  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cfile("hypre_config_test_version.c", execfile=True)
    raise UnableToRunException("Hypre", errormsg=cmdoutput)
  hypre_version = cmdoutput

  remove_cfile("hypre_config_test_version.c", execfile=True)

  return hypre_version

def pkgLibs(compiler=None, cflags=None, sconsEnv=None):
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
    # FIXME: get_compiler() returns a C++ compiler. Do we need a C compiler?
    compiler = get_compiler(sconsEnv)
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s -o a.out %s hypre_config_test_libs.c" % (compiler,cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_libs.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_libs_str, errormsg=cmdoutput)

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cfile("hypre_config_test_libs.c", execfile=True)
    raise UnableToRunException("Hypre", errormsg=cmdoutput)
  blaslapack_str = string.join(string.split(cmdoutput, '\n'))
    
  remove_cfile("hypre_config_test_libs.c", execfile=True)

  if get_architecture() == "darwin":
    libs_str = "-framework vecLib"
  else:
    libs_str = "-L%s %s" % (getAtlasDir(sconsEnv=sconsEnv),blaslapack_str)

  libs_dir = os.path.join(getHypreDir(sconsEnv=sconsEnv),"lib")
  libs_str += " -L%s -lHYPRE" % libs_dir
  
  return libs_str

def pkgCflags(sconsEnv=None):
  include_dir = os.path.join(getHypreDir(sconsEnv=sconsEnv),"include")
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

  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
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

int main(int argc, char *argv[])
{
#ifdef HYPRE_SEQUENTIAL
  MPI_Comm comm = MPI_COMM_NULL;
  HYPRE_IJMatrix A;
  HYPRE_IJMatrixCreate(comm, 0, 10, 0, 10, &A);
#else
  int rank, nprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  HYPRE_IJMatrix A;
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, 10, 0, 10, &A);
  MPI_Finalize();
#endif
  return 0;
} 
"""
  write_cppfile(cpp_testlib_str, "hypre_config_test_libstest.c")

  cmdstr = "%s %s -c hypre_config_test_libstest.c" % (compiler,cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cfile("hypre_config_test_libstest.c")
    raise UnableToCompileException("Hypre", cmd=cmdstr,
                                   program=cpp_testlib_str, errormsg=cmdoutput)
  
  cmdstr = "%s -o a.out %s hypre_config_test_libstest.o" % (linker,libs)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    remove_cfile("hypre_config_test_libstest.c", ofile=True)
    raise UnableToLinkException("Hypre", cmd=cmdstr,
                                program=cpp_testlib_str, errormsg=cmdoutput)

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
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
