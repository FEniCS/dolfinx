#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

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
  
def find_dependency_file(dirs, filename, what="", package=""):
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
  amd_dir = getPackageDir("amd", sconsEnv=sconsEnv, default=None)
  ufconfig_dir = getPackageDir("ufconfig", sconsEnv=sconsEnv, default=None)
  umfpack_dir = getPackageDir("umfpack", sconsEnv=sconsEnv, default=None)
  for d in amd_dir, ufconfig_dir, umfpack_dir:
    if d:
      base_dirs.insert(0, d)
  return base_dirs

def needAMD(sconsEnv):
  """Return True if UMFPACK depends on AMD."""
  umfpack_include_dir = getUmfpackIncDir(sconsEnv)
  need_amd = False
  for line in file(os.path.join(umfpack_include_dir, "umfpack.h")):
    if '#include "amd.h"' in line:
      need_amd = True
      #print "UMFPACK needs AMD"
  return need_amd

def needUFconfig(sconsEnv):
  """Return True if UMFPACK depends on UFconfig."""
  umfpack_include_dir = getUmfpackIncDir(sconsEnv)
  need_ufconfig = False
  for line in file(os.path.join(umfpack_include_dir, "umfpack.h")):
    if '#include "UFconfig.h"' in line:
      need_ufconfig = True
      #print "UMFPACK needs UFconfig"
  return need_ufconfig

def getAMDDirs(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "amd", "AMD", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
  amd_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="amd.h", what="includes", package="AMD")
  try:
    amd_lib_dir = \
        find_dependency_file(all_lib_dirs, filename="libamd.a", what="libs", package="AMD")
  except:
    # Look for shared library libamd.so since we are unable to
    # find static library libamd.a:
    amd_lib_dir = \
        find_dependency_file(all_lib_dirs, filename="libamd.so", what="libs", package="AMD")
  return amd_include_dir, amd_lib_dir

def getAMDIncDir(sconsEnv):
  return getAMDDirs(sconsEnv)[0]

def getAMDLibDir(sconsEnv):
  return getAMDDirs(sconsEnv)[1]

def getUFconfigIncDir(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "ufconfig", "UFconfig", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
  ufconfig_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="UFconfig.h", what="includes", package="UFconfig")
  return ufconfig_include_dir

def getUmfpackDirs(sconsEnv):
  # There are several ways umfpack can be installed:
  # 1. As part of ufsparse (e.g. the ubuntu package). 
  # 2. As separate debian umfpack package
  # 3. From sources, installed manually in some prefix
  # 4. From sources, and not "installed", i.e., left "as is"
  #
  # case 1: includes in /usr/include/ufsparse
  #         libs in /usr
  # case 2: includes in /usr/include/umfpack
  #         libs in /usr
  # case 3: includes in $prefix/umfpack/include or $prefix/include/umfpack
  #                  or $prefix/UMFPACK/include or $prefix/include/UMFPACK
  #         libs in $prefix/umfpack/lib 
  #              or $prefix/UMFPACK/lib
  #         A plain $prefix/{include,lib} is probably also possible.
  # case 4: includes in $prefix/UMFPACK/Include
  #         libs in $prefix/UMFPACK/Lib

  umfpack_include_dir = ""
  umfpack_lib_dir = ""
  base_dirs = getBaseDirs(sconsEnv)

  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
 
  umfpack_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="umfpack.h", what="includes", package="UMFPACK")
  try:
    umfpack_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libumfpack.a", what="libs", package="UMFPACK")
  except UnableToFindPackageException:
    # Look for shared library libumfpack.so since we are unable to
    # find static library libumfpack.a:
    umfpack_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libumfpack.so", what="libs", package="UMFPACK")
  
  return umfpack_include_dir, umfpack_lib_dir

def getUmfpackIncDir(sconsEnv):
  return getUmfpackDirs(sconsEnv)[0]

def getUmfpackLibDir(sconsEnv):
  return getUmfpackDirs(sconsEnv)[1]

def pkgVersion(compiler=None, cflags=None, libs=None, sconsEnv=None):
  # a program to test that we can find umfpack includes
  cpp_test_include_str = r"""
#include <stdio.h>
#include <umfpack.h>

int main() {
  #ifdef UMFPACK_MAIN_VERSION
    #ifdef UMFPACK_SUB_VERSION
      #ifdef UMFPACK_SUBSUB_VERSION
        printf("%d.%d.%d", UMFPACK_MAIN_VERSION,UMFPACK_SUB_VERSION,UMFPACK_SUBSUB_VERSION);
      #else
        printf("%d.%d", UMFPACK_MAIN_VERSION,UMFPACK_SUB_VERSION);
      #endif
    #else
      printf("%d", UMFPACK_MAIN_VERSION);
    #endif
  #endif
  return 0;
}
"""
  write_cppfile(cpp_test_include_str,"umfpack_config_test_include.cpp");

  if not compiler:
    compiler = get_compiler(sconsEnv)
  if not libs:
    libs = pkgLibs()
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s %s umfpack_config_test_include.cpp %s" % (compiler, cflags, libs)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("umfpack_config_test_include.cpp")
    raise UnableToCompileException("UMFPACK", cmd=cmdstr,
                                   program=cpp_test_include_str, errormsg=cmdoutput)
  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cppfile("umfpack_config_test_include.cpp", execfile=True)
    raise UnableToRunException("UMFPACK", errormsg=cmdoutput)
  umfpack_version = cmdoutput

  remove_cppfile("umfpack_config_test_include.cpp", execfile=True)
  
  return umfpack_version

def pkgCflags(sconsEnv=None):
  cflags = "-I%s" % getUmfpackIncDir(sconsEnv)
  if needAMD(sconsEnv):
    cflags += " -I%s" % getAMDIncDir(sconsEnv)
  if needUFconfig(sconsEnv):
    cflags += " -I%s" % getUFconfigIncDir(sconsEnv)
  return cflags

def pkgLibs(sconsEnv=None):
  libs = ""
  if get_architecture() == "darwin":
    libs += "-framework vecLib"
  else:
    libs += "-L%s -lblas" % getAtlasDir(sconsEnv=sconsEnv)
  libs += " -L%s -lumfpack" % getUmfpackLibDir(sconsEnv)
  if needAMD(sconsEnv):
    libs += " -L%s -lamd" % getAMDLibDir(sconsEnv)
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
    version = pkgVersion(compiler=compiler, cflags=cflags, libs=libs, sconsEnv=sconsEnv)

  # a program that do a real umfpack test.
  cpp_test_lib_str = r"""
/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) Timothy A. Davis, CISE,                              */
/* Univ. of Florida.  All Rights Reserved.  See ../Doc/License for License.   */
/* web: http://www.cise.ufl.edu/research/sparse/umfpack                       */
/* -------------------------------------------------------------------------- */

/* We reuse the Demo/umfpack_simple.c from the UMFPACK distro                 */

#include <stdio.h>
#include "umfpack.h"

int    n = 5 ;
int    Ap [ ] = {0, 2, 5, 9, 10, 12} ;
int    Ai [ ] = { 0,  1,  0,   2,  4,  1,  2,  3,   4,  2,  1,  4} ;
double Ax [ ] = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.} ;
double b [ ] = {8., 45., -3., 3., 19.} ;
double x [5] ;

int main (void)
{
    double *null = (double *) NULL ;
    int i ;
    void *Symbolic, *Numeric ;
    (void) umfpack_di_symbolic (n, n, Ap, Ai, Ax, &Symbolic, null, null) ;
    (void) umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, null, null) ;
    umfpack_di_free_symbolic (&Symbolic) ;
    (void) umfpack_di_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, null, null) ;
    umfpack_di_free_numeric (&Numeric) ;
    for (i = 0 ; i < n ; i++) printf ("x [%d] = %g\n", i, x [i]) ;
    return (0) ;
}
"""
  write_cppfile(cpp_test_lib_str,"umfpack_config_test_lib.cpp");

  # try to compile the simple umfpack test
  cmdstr = "%s %s -c umfpack_config_test_lib.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile("umfpack_config_test_lib.cpp")
    raise UnableToCompileException("UMFPACK", cmd=cmdstr,
                                   program=cpp_test_lib_str, errormsg=cmdoutput)

  cmdstr = "%s umfpack_config_test_lib.o %s" % (linker, libs)
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile("umfpack_config_test_lib.cpp", ofile=True)
    errormsg = ("Using '%s' for BLAS, consider setting the environment " + \
                "variable ATLAS_DIR if this is wrong.\n") % getAtlasDir()
    errormsg += cmdoutput
    raise UnableToLinkException("UMFPACK", cmd=cmdstr,
                                program=cpp_test_lib_str, errormsg=errormsg)

  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    remove_cppfile("umfpack_config_test_lib.cpp", ofile=True, execfile=True)
    raise UnableToRunException("UMFPACK", errormsg=cmdoutput)
  cmdlines = cmdoutput.split("\n")
  values = []
  for line in cmdlines:
    values.append(int(line.split(" ")[3]))
  if not values == [1, 2, 3, 4, 5]:
    errormsg = "UMFPACK test does not produce correct result, check your UMFPACK installation."
    errormsg += "\n%s" % cmdoutput
    raise UnableToRunException("UMFPACK", errormsg=errormsg)

  remove_cppfile("umfpack_config_test_lib.cpp", ofile=True, execfile=True)

  return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: UMFPACK
Version: %s
Description: The UMFPACK project, a set of routines for solving sparse linear systems via LU factorization.
Libs: %s
Cflags: %s
""" % (version, libs, cflags)
  pkg_file = open(os.path.join(directory,"umfpack.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found UMFPACK and generated pkg-config file in\n '%s'" % directory

if __name__ == "__main__": 
  generatePkgConf(directory=".")
