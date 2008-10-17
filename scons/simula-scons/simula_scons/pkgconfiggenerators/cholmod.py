#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def generate_dirs(base_dirs, *names):
  include_exts = [["include"], ["Include"]]
  for name in names:
    include_exts.append([name])
    include_exts.append(["include", name])
    include_exts.append([name, "include"])
    include_exts.append([name, "Include"])
  include_exts = [os.path.join(*d) for d in include_exts]
  lib_exts = [["lib"], ["Lib"]]
  for name in names:
    lib_exts.append([name])
    lib_exts.append(["lib", name])
    lib_exts.append([name, "lib"])
    lib_exts.append([name, "Lib"])
  lib_exts = [os.path.join(*d) for d in lib_exts]

  all_include_dirs = \
      [os.path.join(a,b) for a in base_dirs for b in include_exts]
  # move the most likely ones to the end
  all_include_dirs.reverse()
  all_lib_dirs = [os.path.join(a,b) for a in base_dirs for b in lib_exts]
  # move the most likely ones to the end
  all_lib_dirs.reverse()

  return (all_include_dirs, all_lib_dirs)
  
def find_dependency_file(dirs, filename, package=""):
  found_file = False
  found_dir = ""
  for d in dirs:
    found_dir = d
    if os.path.isfile(os.path.join(found_dir, filename)):
      found_file = True
      break
  if not found_file:
    raise UnableToFindPackageException(package)
  return found_dir

def getBaseDirs(sconsEnv):
  base_dirs = \
      [os.path.join(os.path.sep, *d) for d in [["usr"], ["usr", "local"]]]
  if get_architecture() == 'darwin':
    # use fink as default
    base_dirs.append(os.path.join(os.path.sep,"sw"))
  cholmod_dir = getPackageDir("cholmod", sconsEnv=sconsEnv, default='/usr')
  base_dirs.insert(0, cholmod_dir)
  return base_dirs

def getColamdDirs(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", "colamd", "COLAMD")
  colamd_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="colamd.h", package="COLAMD")
  try:
    colamd_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libcolamd.a", package="COLAMD")
  except:
    # Look for shared library libamd.so since we are unable to
    # find static library libamd.a:
    colamd_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libcolamd.so", package="COLAMD")
  return colamd_include_dir, colamd_lib_dir

def getColamdIncDir(sconsEnv):
  return getColamdDirs(sconsEnv)[0]

def getColamdLibDir(sconsEnv):
  return getColamdDirs(sconsEnv)[1]

def getAmdDirs(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", "amd", "AMD")
  amd_include_dir = \
      find_dependency_file(all_include_dirs, filename="amd.h", package="AMD")
  try:
    amd_lib_dir = \
        find_dependency_file(all_lib_dirs, filename="libamd.a", package="AMD")
  except:
    # Look for shared library libamd.so since we are unable to
    # find static library libamd.a:
    amd_lib_dir = \
        find_dependency_file(all_lib_dirs, filename="libamd.so", package="AMD")
  return amd_include_dir, amd_lib_dir

def getAmdIncDir(sconsEnv):
  return getAmdDirs(sconsEnv)[0]

def getAmdLibDir(sconsEnv):
  return getAmdDirs(sconsEnv)[1]

def getUFconfigIncDir(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", "ufconfig", "UFconfig")
  ufconfig_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="UFconfig.h", package="UFconfig")
  return ufconfig_include_dir

def getCholmodDirs(sconsEnv):
  # There are several ways CHOLMOD can be installed:
  # 1. As part of suitesparse/ufsparse (e.g. the ubuntu package). 
  # 2. As separate debian CHOLMOD package
  # 3. From sources, installed manually in some prefix
  # 4. From sources, and not "installed", i.e., left "as is"
  #
  # case 1: includes in /usr/include/suitesparse or /usr/include/ufsparse
  #                  or /usr/include/UFSPARSE
  #         libs in /usr
  # case 2: includes in /usr/include/cholmod
  #         libs in /usr
  # case 3: includes in $prefix/cholmod/include or $prefix/include/cholmod
  #                  or $prefix/CHOLMOD/include or $prefix/include/CHOLMOD
  #         libs in $prefix/cholmod/lib 
  #              or $prefix/CHOLMOD/lib
  #         A plain $prefix/{include,lib} is probably also possible.
  # case 4: includes in $prefix/CHOLMOD/Include
  #         libs in $prefix/CHOLMOD/Lib
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs,
                    "suitesparse", "ufsparse", "UFSPARSE", "cholmod", "CHOLMOD")
  cholmod_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="cholmod.h", package="CHOLMOD")
  try:
    cholmod_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libcholmod.a", package="CHOLMOD")
  except:
    # Look for shared library libcholmod.so since we are unable to
    # find static library libcholmod.a:
    cholmod_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libcholmod.so", package="AMD")
  return cholmod_include_dir, cholmod_lib_dir

def getCholmodIncDir(sconsEnv):
  return getCholmodDirs(sconsEnv)[0]

def getCholmodLibDir(sconsEnv):
  return getCholmodDirs(sconsEnv)[1]

def pkgVersion(compiler=None, cflags=None, libs=None, sconsEnv=None):
  # a program to test that we can find CHOLMOD includes
  cpp_test_include_str = r"""
#include <stdio.h>
#include <cholmod.h>

int main() {
  #ifdef CHOLMOD_MAIN_VERSION
    #ifdef CHOLMOD_SUB_VERSION
      #ifdef CHOLMOD_SUBSUB_VERSION
        printf("%d.%d.%d", CHOLMOD_MAIN_VERSION,CHOLMOD_SUB_VERSION,CHOLMOD_SUBSUB_VERSION);
      #else
        printf("%d.%d", CHOLMOD_MAIN_VERSION,CHOLMOD_SUB_VERSION);
      #endif
    #else
      printf("%d", CHOLMOD_MAIN_VERSION);
    #endif
  #endif
  return 0;
}
"""
  write_cppfile(cpp_test_include_str, "cholmod_config_test_include.cpp");

  if not compiler:
    compiler = get_compiler(sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv)
  if not cflags:
    cflags = pkgCflags(sconsEnv)
  cmdstr = "%s -o a.out %s cholmod_config_test_include.cpp %s" % \
           (compiler, cflags, libs)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    # Try adding -lgfortran so get around Ubuntu Hardy libatlas-base-dev issue
    libs += " -lgfortran"
    cmdstr = "%s -o a.out %s cholmod_config_test_include.cpp %s" % \
             (compiler, cflags, libs)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
      remove_cppfile("cholmod_config_test_include.cpp")
      raise UnableToCompileException("CHOLMOD", cmd=cmdstr,
                                     program=cpp_test_include_str,
                                     errormsg=cmdoutput)
  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cppfile("cholmod_config_test_include.cpp", execfile=True)
    raise UnableToRunException("CHOLMOD", errormsg=cmdoutput)
  cholmod_version = cmdoutput

  remove_cppfile("cholmod_config_test_include.cpp", execfile=True)
  
  return cholmod_version

def pkgCflags(sconsEnv=None):
  cflags = ""
  for inc_dir in set([getCholmodIncDir(sconsEnv), getAmdIncDir(sconsEnv),
                      getColamdIncDir(sconsEnv), getUFconfigIncDir(sconsEnv)]):
    cflags += " -I%s" % inc_dir
  return cflags.strip()

def pkgLibs(sconsEnv=None):
  libs = ""
  if get_architecture() == "darwin":
    libs += "-framework vecLib"
  else:
    libs += "-L%s -llapack -L%s -lblas" % \
            (getLapackDir(sconsEnv=sconsEnv), getBlasDir(sconsEnv=sconsEnv))
  libs += " -L%s -lcholmod" % getCholmodLibDir(sconsEnv)
  libs += " -L%s -lamd" % getAmdLibDir(sconsEnv)
  libs += " -L%s -lcolamd" % getColamdLibDir(sconsEnv)
  return libs

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
  """Run the tests for this package
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both
  """
  # Set architecture up front.
  arch = get_architecture()

  # set which compiler and linker to use:
  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)
  if not libs:
    libs = pkgLibs(sconsEnv=sconsEnv)
  if not version:
    version = pkgVersion(compiler=compiler, cflags=cflags,
                         libs=libs, sconsEnv=sconsEnv)

  # FIXME: add a program that do a real CHOLMOD test.
  cpp_test_lib_str = r"""
#include "cholmod.h"
int main (void)
{
    cholmod_sparse *A ;
    return (0) ;
}
"""
  write_cppfile(cpp_test_lib_str,"cholmod_config_test_lib.cpp");

  # try to compile the simple CHOLMOD test
  cmdstr = "%s %s -c cholmod_config_test_lib.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("cholmod_config_test_lib.cpp")
    raise UnableToCompileException("CHOLMOD", cmd=cmdstr,
                                   program=cpp_test_lib_str, errormsg=cmdoutput)

  cmdstr = "%s -o a.out %s cholmod_config_test_lib.o" % (linker, libs)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # Try adding -lgfortran so get around Ubuntu Hardy libatlas-base-dev issue
    libs += " -lgfortran"
    cmdstr = "%s -o a.out %s cholmod_config_test_lib.o" % (linker, libs)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
      remove_cppfile("cholmod_config_test_lib.cpp", ofile=True)
      errormsg = ("Using '%s' for LAPACK and '%s' BLAS. Consider setting the " \
                  "environment variables LAPACK_DIR and BLAS_DIR if this is " \
                  "wrong.\n") % (getLapackDir(sconsEnv), getBlasDir(sconsEnv))
      errormsg += cmdoutput
      raise UnableToLinkException("CHOLMOD", cmd=cmdstr,
                                  program=cpp_test_lib_str, errormsg=errormsg)

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cppfile("cholmod_config_test_lib.cpp", ofile=True, execfile=True)
    raise UnableToRunException("CHOLMOD", errormsg=cmdoutput)

  remove_cppfile("cholmod_config_test_lib.cpp", ofile=True, execfile=True)

  return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: CHOLMOD
Version: %s
Description: CHOLMOD is a set of ANSI C routines for sparse Cholesky factorization and update/downdate.
Libs: %s
Cflags: %s
""" % (version, libs, cflags)
  pkg_file = open(os.path.join(directory, "cholmod.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found CHOLMOD and generated pkg-config file in\n '%s'" \
        % directory

if __name__ == "__main__": 
  generatePkgConf(directory=".")
