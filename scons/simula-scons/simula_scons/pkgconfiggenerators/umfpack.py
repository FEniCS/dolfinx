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
  while not found_file and dirs:
    found_dir = dirs.pop()
    if os.path.isfile(os.path.join(found_dir, filename)):
      found_file = True
  if not found_file:
    raise UnableToFindPackage(package)
  return found_dir

def getBaseDirs():
  base_dirs = \
      [os.path.join(os.path.sep, *d) for d in [["usr"], ["usr", "local"]]]
  if get_architecture() == 'darwin':
    # use fink as default
    base_dirs.append(os.path.join(os.path.sep,"sw"))
  if os.environ.has_key("UMFPACK_DIR"):
    base_dirs.insert(0, os.environ["UMFPACK_DIR"])
  # FIXME: should we also look for AMD_DIR and UFCONFIG_DIR in os.environ?
  return base_dirs

def needAMD():
  """Return True if UMFPACK depends on AMD."""
  umfpack_include_dir = getUmfpackIncDir()
  need_amd = False
  for line in file(os.path.join(umfpack_include_dir, "umfpack.h")):
    if '#include "amd.h"' in line:
      need_amd = True
      #print "UMFPACK needs AMD"
  return need_amd

def needUFconfig():
  """Return True if UMFPACK depends on UFconfig."""
  umfpack_include_dir = getUmfpackIncDir()
  need_ufconfig = False
  for line in file(os.path.join(umfpack_include_dir, "umfpack.h")):
    if '#include "UFconfig.h"' in line:
      need_ufconfig = True
      #print "UMFPACK needs UFconfig"
  return need_ufconfig

def getAMDDirs():
  base_dirs = getBaseDirs()
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "amd", "AMD", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
  amd_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="amd.h", what="includes", package="AMD")
  amd_lib_dir = \
      find_dependency_file(all_lib_dirs, filename="libamd.a", what="libs", package="AMD")
  return amd_include_dir, amd_lib_dir

def getAMDIncDir():
  return getAMDDirs()[0]

def getAMDLibDir():
  return getAMDDirs()[1]

def getUFconfigIncDir():
  base_dirs = getBaseDirs()
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "ufconfig", "UFconfig", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
  ufconfig_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="UFconfig.h", what="includes", package="UFconfig")
  return ufconfig_include_dir

def getUmfpackDirs():
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
  base_dirs = getBaseDirs()

  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse", "ufsparse", "UFSPARSE", "umfpack", "UMFPACK")
 
  umfpack_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="umfpack.h", what="includes", package="UMFPACK")
  umfpack_lib_dir = \
      find_dependency_file(all_lib_dirs,
                           filename="libumfpack.a", what="libs", package="UMFPACK")
  
  return umfpack_include_dir, umfpack_lib_dir

def getUmfpackIncDir():
  return getUmfpackDirs()[0]

def getUmfpackLibDir():
  return getUmfpackDirs()[1]

def pkgVersion(compiler=None, cflags=None, libs=None, **kwargs):
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
    compiler = get_compiler(kwargs.get("sconsEnv", None))
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

def pkgCflags(**kwargs):
  cflags = "-I%s" % getUmfpackIncDir()
  if needAMD():
    cflags += " -I%s" % getAMDIncDir()
  if needUFconfig():
    cflags += " -I%s" % getUFconfigIncDir()
  return cflags

def getATLASDir():
  # Set ATLAS location from environ, or use default
  if os.environ.has_key('ATLAS_DIR'):
    atlaslocation = os.environ["ATLAS_DIR"]
  else:
    atlaslocation = os.path.join(os.path.sep,"usr","lib","atlas")
  return atlaslocation

def pkgLibs(**kwargs):
  libs = ""
  if get_architecture() == "darwin":
    libs += "-framework vecLib"
  else:
    libs += "-L%s -lblas" % getATLASDir()
  libs += " -L%s -lumfpack" % getUmfpackLibDir()
  if needAMD():
    libs += " -L%s -lamd" % getAMDLibDir() 
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
    cflags = pkgCflags()
  if not libs:
    libs = pkgLibs()
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

  cmdstr = "%s %s umfpack_config_test_lib.o" % (linker, libs)
  linkFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if linkFailed:
    errormsg = ("Using '%s' for BLAS, consider setting the environment " + \
                "variable ATLAS_DIR if this is wrong.\n") % getATLASDir()
    errormsg += cmdoutput
    raise UnableToLinkException("UMFPACK", cmd=cmdstr,
                                program=cpp_test_lib_str, errormsg=errormsg)

  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
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
