#!/usr/bin/env python

import os

from commonPkgConfigUtils import *

def pkgVersion(compiler=None, linker=None,
               cflags=None, libs=None, sconsEnv=None):
    # FIXME: Not sure how to extract version information from LAPACK
    return "3.0"

def pkgCflags(sconsEnv=None):
    return ""

def pkgLibs(compiler=None, linker=None, cflags=None, sconsEnv=None):
    # A simple program that do a real LAPACK test (thanks to Anders Logg)
    cpp_test_lib_str = r"""
extern "C"
{
  void dgels_(char* trans, int* m, int* n, int* nrhs,
              double* a, int* lda, double* b, int* ldb,
              double* work, int* lwork, int* info);
}

int main()
{
  double A[4] = {1, 0, 0, 1};
  double b[2] = {1, 1};

  char trans = 'N';
  int m = 2;
  int n = 2;
  int nrhs = 1;
  int lda = 2;
  int ldb = 2;
  int lwork = 4;
  int status = 0;
  double* work = new double[m*lwork];

  dgels_(&trans, &m, &n, &nrhs,
         A, &lda, b, &ldb,
         work, &lwork, &status);

  delete [] work;

  return status;
}
"""
    cpp_file = "lapack_config_test_lib.cpp"
    write_cppfile(cpp_test_lib_str, cpp_file);

    if not compiler:
        compiler = get_compiler(sconsEnv)
    if not linker:
        linker = get_linker(sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)

    # try to compile and run the LAPACK test program
    cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("LAPACK", cmd=cmdstr,
                                       program=cpp_test_lib_str,
                                       errormsg=cmdoutput)
    
    if get_architecture() == "darwin":
        libs = "-framework vecLib"
    else:
        libs = "-L%s -llapack -L%s -lblas" % \
               (getLapackDir(sconsEnv), getBlasDir(sconsEnv))

    cmdstr = "%s -o a.out %s %s" % \
             (linker, libs, cpp_file.replace('.cpp', '.o'))
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
        # try adding -lgfortran to get around Hardy libatlas-base-dev issue
        libs += " -lgfortran"
        cmdstr = "%s -o a.out %s %s" % \
                 (linker, libs, cpp_file.replace('.cpp', '.o'))
        linkFailed, cmdoutput = getstatusoutput(cmdstr)
        if linkFailed:
            remove_cppfile(cpp_file, ofile=True)
            errormsg = """Using '%s' for LAPACK and '%s' BLAS.
Consider setting the environment variables LAPACK_DIR and
BLAS_DIR if this is wrong.

%s
""" % (getLapackDir(sconsEnv), getBlasDir(sconsEnv), cmdoutput)
            raise UnableToLinkException("LAPACK", cmd=cmdstr,
                                        program=cpp_test_lib_str,
                                        errormsg=errormsg)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile(cpp_file, ofile=True, execfile=True)
        raise UnableToRunException("LAPACK", errormsg=cmdoutput)

    remove_cppfile(cpp_file, ofile=True, execfile=True)

    return libs

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
    """Run the tests for this package
     
    If Ok, return various variables, if not we will end with an exception.
    forceCompiler, if set, should be a tuple containing (compiler, linker)
    or just a string, which in that case will be used as both
    """

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
        libs = pkgLibs(compiler=compiler, linker=linker,
                       cflags=cflags, sconsEnv=sconsEnv)
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv, compiler=compiler,
                             linker=linker, cflags=cflags, libs=libs)

    return version, cflags, libs

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):
    if directory is None:
        directory = suitablePkgConfDir()
    
    version, cflags, libs = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: LAPACK
Version: %s
Description: Linear Algebra PACKage
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])

    pkg_file = open(os.path.join(directory, "lapack.pc"), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()
    print "done\n Found LAPACK and generated pkg-config file in\n '%s'" \
          % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")
