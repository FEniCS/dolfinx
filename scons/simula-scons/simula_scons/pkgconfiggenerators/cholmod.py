#!/usr/bin/env python
import os,sys
import string
import os.path

import lapack

from commonPkgConfigUtils import *

def generate_dirs(base_dirs, *names):
  include_exts = [["include"], ["Include"]]
  for name in names:
    include_exts.append([name])
    include_exts.append(["include", name])
    include_exts.append([name, "include"])
    include_exts.append([name, "Include"])
    include_exts.append([name, "Lib"])  # metis.h could be in METIS_DIR/Lib
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
  all_lib_dirs = [os.path.join(a,b) for a in base_dirs for b in lib_exts]

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
  cholmod_dir = getPackageDir("cholmod", sconsEnv=sconsEnv, default=None)
  deps = [getPackageDir("amd", sconsEnv=sconsEnv, default=cholmod_dir),
          getPackageDir("camd", sconsEnv=sconsEnv, default=cholmod_dir),
          getPackageDir("colamd", sconsEnv=sconsEnv, default=cholmod_dir),
          getPackageDir("ccolamd", sconsEnv=sconsEnv, default=cholmod_dir),
          getPackageDir("ufconfig", sconsEnv=sconsEnv, default=cholmod_dir)]
  for d in set([cholmod_dir] + deps):
    if d:
      base_dirs.insert(0, d)
  return base_dirs

def getPackageDirs(sconsEnv, package):
  # There are several ways package Foo can be installed:
  # 1. As part of suitesparse/ufsparse (e.g. the ubuntu package). 
  # 2. As separate debian Foo package
  # 3. From sources, installed manually in some prefix
  # 4. From sources, and not "installed", i.e., left "as is"
  #
  # case 1: includes in /usr/include/suitesparse or /usr/include/ufsparse
  #                  or /usr/include/UFSPARSE
  #         libs in /usr
  # case 2: includes in /usr/include/cholmod
  #         libs in /usr
  # case 3: includes in $prefix/foo/include or $prefix/include/foo
  #                  or $prefix/FOO/include or $prefix/include/FOO
  #         libs in $prefix/foo/lib 
  #              or $prefix/FOO/lib
  #         A plain $prefix/{include,lib} is probably also possible.
  # case 4: includes in $prefix/FOO/Include
  #         libs in $prefix/FOO/Lib
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", package, package.lower())
  package_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="%s.h" % package.lower(),
                           package=package)
  try:
    package_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="lib%s.a" % package.lower(),
                             package=package)
  except:
    # Look for shared library libfoo.so since we are unable to
    # find static library libfoo.a:
    package_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="lib%s.so" % package.lower(),
                             package=package)
  return package_include_dir, package_lib_dir

def getUFconfigIncDir(sconsEnv):
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", "ufconfig", "UFconfig")
  ufconfig_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="UFconfig.h", package="UFconfig")
  return ufconfig_include_dir

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

def getMETISDirs(sconsEnv):
  # There are several ways METIS can be installed:
  # 1. As part of suitesparse/ufsparse (e.g. the ubuntu package). 
  # 2. As separate debian Foo package
  # 3. From sources, installed manually in some prefix
  # 4. From sources, and not "installed", i.e., left "as is"
  #
  # case 1: includes in /usr/include/suitesparse or /usr/include/ufsparse
  #                  or /usr/include/UFSPARSE
  #         libs in /usr
  # case 2: includes in /usr/include/cholmod
  #         libs in /usr
  # case 3: includes in $prefix/foo/include or $prefix/include/foo
  #                  or $prefix/FOO/include or $prefix/include/FOO
  #         libs in $prefix/foo/lib 
  #              or $prefix/FOO/lib
  #         A plain $prefix/{include,lib} is probably also possible.
  # case 4: includes in $prefix/FOO/Include
  #         libs in $prefix/FOO/Lib
  base_dirs = getBaseDirs(sconsEnv)
  all_include_dirs, all_lib_dirs = \
      generate_dirs(base_dirs, "suitesparse",
                    "ufsparse", "UFSPARSE", "metis-4.0", "metis")
  metis_include_dir = \
      find_dependency_file(all_include_dirs,
                           filename="metis.h", package="METIS")
  try:
    metis_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libmetis.a", package="METIS")
  except UnableToFindPackageException:
    # Look for shared library libmetis.so since we are unable to
    # find static library libmetis.a:
    metis_lib_dir = \
        find_dependency_file(all_lib_dirs,
                             filename="libmetis.so", package="METIS")
  return metis_include_dir, metis_lib_dir

def pkgCflags(sconsEnv=None):
  # CHOLMOD relies on several other packages: AMD, CAMD, COLAMD, CCOLAMD,
  # UFconfig, METIS, the BLAS, and LAPACK. All but METIS, the BLAS, and
  # LAPACK are part of SuiteSparse. Only METIS is optional.
  cflags = ""
  for inc_dir in set([getPackageDirs(sconsEnv, "AMD")[0],
                      getPackageDirs(sconsEnv, "CAMD")[0],
                      getPackageDirs(sconsEnv, "COLAMD")[0],
                      getPackageDirs(sconsEnv, "CCOLAMD")[0],
                      getUFconfigIncDir(sconsEnv),
                      getPackageDirs(sconsEnv, "CHOLMOD")[0]]):
    cflags += " -I%s" % inc_dir

  return cflags.strip()

def pkgLibs(compiler=None, linker=None, cflags=None, sconsEnv=None):
  if not compiler:
    compiler = get_compiler(sconsEnv)
  if not linker:
    linker = get_linker(sconsEnv)
  if not cflags:
    cflags = pkgCflags(sconsEnv=sconsEnv)

  # create a simple test program that uses CHOLMOD
  cpp_test_libs_str = r"""
#include <stdio.h>
#include <cholmod.h>

int main()
{
  cholmod_dense *D;
  cholmod_sparse *S;
  cholmod_dense *x, *b, *r;
  cholmod_factor *L;
  double one[2] = {1,0}, m1[2] = {-1,0};
  double *dx;
  cholmod_common c;
  int n = 5;
  double K[5][5] = {{1.0, 0.0, 0.0, 0.0, 0.0},
		    {0.0, 2.0,-1.0, 0.0, 0.0},
		    {0.0,-1.0, 2.0,-1.0, 0.0},
		    {0.0, 0.0,-1.0, 2.0, 0.0},
		    {0.0, 0.0, 0.0, 0.0, 1.0}};
  cholmod_start (&c);
  D = cholmod_allocate_dense(n, n, n, CHOLMOD_REAL, &c);
  dx = (double*)D->x;
  for (int i=0; i < n; i++)
    for (int j=0; j < n; j++)
      dx[i+j*n] = K[i][j];
  S = cholmod_dense_to_sparse(D, 1, &c);
  S->stype = 1;
  cholmod_reallocate_sparse(cholmod_nnz(S, &c), S, &c);
  b = cholmod_ones(S->nrow, 1, S->xtype, &c);
  L = cholmod_analyze(S, &c);
  cholmod_factorize(S, L, &c);
  x = cholmod_solve(CHOLMOD_A, L, b, &c);
  r = cholmod_copy_dense(b, &c);
  cholmod_sdmult(S, 0, m1, one, x, r, &c);
  printf("norm(b-Ax)=%g\n", cholmod_norm_dense(r, 0, &c));
  cholmod_free_factor(&L, &c);
  cholmod_free_dense(&D, &c);
  cholmod_free_sparse(&S, &c);
  cholmod_free_dense(&r, &c);
  cholmod_free_dense(&x, &c);
  cholmod_free_dense(&b, &c);
  cholmod_finish(&c);
  return 0;
}
"""
  cpp_file = "cholmod_test_libs.cpp"
  write_cppfile(cpp_test_libs_str, cpp_file);

  libs = ""
  for lib_dir in set([getPackageDirs(sconsEnv, "AMD")[1],
                      getPackageDirs(sconsEnv, "CAMD")[1],
                      getPackageDirs(sconsEnv, "COLAMD")[1],
                      getPackageDirs(sconsEnv, "CCOLAMD")[1],
                      getPackageDirs(sconsEnv, "CHOLMOD")[1]]):
    libs += " -L%s" % lib_dir
  libs += " -lcholmod -lamd -lcamd -lcolamd -lccolamd"

  # test that we can compile
  lapack_cflags = lapack.pkgCflags(sconsEnv=sconsEnv)
  cmdstr = "%s %s %s -c %s" % (compiler, cflags, lapack_cflags, cpp_file)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile(cpp_file)
    raise UnableToCompileException("CHOLMOD", cmd=cmdstr,
                                   program=cpp_test_libs_str,
                                   errormsg=cmdoutput)

  # test that we can link
  lapack_libs = lapack.pkgLibs(compiler=compiler, linker=linker,
                               cflags=cflags, sconsEnv=sconsEnv)
  cmdstr = "%s -o a.out %s %s %s" % \
           (linker, cpp_file.replace('.cpp', '.o'), libs, lapack_libs)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # CHOLMOD may be built with METIS, try adding -lmetis
    metis_dir = getMETISDirs(sconsEnv)[0]
    cmdstr = "%s -o a.out %s %s %s -L%s -lmetis" % \
             (linker, cpp_file.replace('.cpp', '.o'),
              libs, lapack_libs, metis_dir)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:    
      remove_cppfile(cpp_file, ofile=True)
      errormsg = """Using '%s' for METIS.
Consider setting the environment variable METIS_DIR if this is
wrong. This is not necessary if CHOLMOD is built without METIS.

%s
""" % (metis_dir, cmdoutput)
      raise UnableToCompileException("CHOLMOD", cmd=cmdstr,
                                     program=cpp_test_libs_str,
                                     errormsg=errormsg)
    else:
      libs += " -L%s -lmetis" % metis_dir

  cmdstr = os.path.join(os.getcwd(), "a.out")
  runFailed, cmdoutput = getstatusoutput(cmdstr)
  if runFailed:
    remove_cppfile(cpp_file, execfile=True)
    raise UnableToRunException("CHOLMOD", errormsg=cmdoutput)
  value = int(cmdoutput.split('=')[1])
  if value != 0:
    errormsg = "CHOLMOD test does not produce correct result, " \
               "check your CHOLMOD installation."
    errormsg += "\n%s" % cmdoutput
    raise UnableToRunException("CHOLMOD", errormsg=errormsg)

  remove_cppfile(cpp_file, ofile=True, execfile=True)

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
  else:
    # run pkgLibs as this is the current CHOLMOD test
    pkgLibs(compiler=compiler, linker=linker,
            cflags=cflags, sconsEnv=sconsEnv) 
  if not version:
    version = pkgVersion(compiler=compiler, cflags=cflags,
                         libs=libs, sconsEnv=sconsEnv)

  return version, libs, cflags

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):
  if directory is None:
    directory = suitablePkgConfDir()

  # cholmod.pc requires lapack.pc so make sure it is available
  dep_module_header_and_libs('LAPACK', 'lapack', sconsEnv=sconsEnv)

  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: CHOLMOD
Version: %s
Description: CHOLMOD is a set of ANSI C routines for sparse Cholesky factorization and update/downdate.
Requires: lapack
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
  pkg_file = open(os.path.join(directory, "cholmod.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found CHOLMOD and generated pkg-config file in\n '%s'" \
        % directory

if __name__ == "__main__": 
  generatePkgConf(directory=".")
