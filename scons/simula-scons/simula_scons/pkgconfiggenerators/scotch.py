#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getScotchDir(sconsEnv=None):
    default = os.path.join(os.path.sep, "usr")
    scotch_dir = getPackageDir("scotch", sconsEnv=sconsEnv, default=default)
    return scotch_dir

def pkgVersion(sconsEnv=None):
    # Try to figure out the SCOTCH version from scotch.h
    # Fource places to look for scotch.h:
    # 1. SCOTCH_DIR/scotch.h
    # 2. SCOTCH_DIR/bin/scotch.h
    # 3. SCOTCH_DIR/include/scotch.h
    # 4. SCOTCH_DIR/include/scotch/scotch.h
    
    scotch_dir = getScotchDir(sconsEnv=sconsEnv)

    failure = True
    include_dirs = ['', 'bin', 'include', os.path.join('include', 'scotch')]
    for inc_dir in include_dirs:
        scotch_h_filename = os.path.join(scotch_dir, inc_dir, 'scotch.h')
        try:
            scotch_h_file = open(scotch_h_filename, 'r')
        except Exception:
            continue
        failure = False
        break

    if failure:
        msg = "Unable to locate scotch.h. Is SCOTCH_DIR set correctly?"
        raise UnableToXXXException(msg)
    
    scotch_version = '4.0'  # assume 4.0 as default
    for line in scotch_h_file:
        if "Version" in line:
            # the lines with the version number is something like this:
            # /**                # Version 4.0  : from : 11 dec 2001     **/
            tmp = line.split()
            scotch_version = tmp[tmp.index("Version")+1]

    return scotch_version

def pkgCflags(compiler=None, sconsEnv=None):
    # Four places to look for scotch.h:
    # 1. SCOTCH_DIR/scotch.h
    # 2. SCOTCH_DIR/bin/scotch.h
    # 3. SCOTCH_DIR/include/scotch.h
    # 4. SCOTCH_DIR/include/scotch/scotch.h

    if not compiler:
        compiler = get_compiler(sconsEnv)

    scotch_dir = getScotchDir(sconsEnv=sconsEnv)
    
    # Simple test-program that try to include the main scotch header
    cpp_test_include_str = r"""
extern "C" {
#include <stdio.h>
#include <scotch.h>
}
int main() {
  return 0;
}
"""
    cpp_file = "scotch_config_test_include.cpp"
    write_cppfile(cpp_test_include_str, cpp_file)

    include_dirs = ['', 'bin', 'include', os.path.join('include', 'scotch')]
    for inc_dir in include_dirs:
        scotch_inc_dir = os.path.join(scotch_dir, inc_dir)

        cmdstr = "%s -I%s -c %s" % (compiler, scotch_inc_dir, cpp_file)
        compileFailed, cmdoutput = getstatusoutput(cmdstr)
        if not compileFailed:
            break

    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("SCOTCH", cmd=cmdstr,
                                       program=cpp_test_include_str,
                                       errormsg=cmdoutput)

    remove_cppfile(cpp_file, ofile=True)

    return "-I%s" % scotch_inc_dir

def pkgLibs(compiler=None, linker=None, cflags=None, sconsEnv=None):
    # Three cases for libscotch.a:
    # 1. SCOTCH_DIR/libscotch.a
    # 2. SCOTCH_DIR/bin/libscotch.a
    # 3. SCOTCH_DIR/lib/libscotch.a
    # FIXME: should we look for libscotch.so too?
    
    if not compiler:
        compiler = get_compiler(sconsEnv)
    if not linker:
        linker = get_linker(sconsEnv)
    if not cflags:
        cflags = pkgCflags(compiler=compiler, sconsEnv=sconsEnv)

    scotch_dir = getScotchDir(sconsEnv=sconsEnv)
    
    # Test that we can call a SCOTCH function
    cpp_test_lib_str = r"""
extern "C" {
#include <stdio.h>
#include <scotch.h>
}
#include <iostream>
int main() {
  SCOTCH_Mesh mesh;

  if (SCOTCH_meshInit (&mesh) !=0) {
    std::cout << "failure";
  } else {
    std::cout << "success";
  }
  return 0;
}
"""
    cpp_file = "scotch_config_test_lib.cpp"
    write_cppfile(cpp_test_lib_str, cpp_file)

    # test that we can compile a program using SCOTCH:
    cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("SCOTCH", cmd=cmdstr,
                                       program=cpp_test_lib_str,
                                       errormsg=cmdoutput)

    app = os.path.join(os.getcwd(), "a.out")
    lib_dirs = ['', 'bin', 'lib']
    for lib_dir in lib_dirs:
        scotch_lib_dir = os.path.join(scotch_dir, lib_dir)
        libscotch_static = os.path.join(scotch_lib_dir, 'libscotch.a')
        libscotcherr_static = os.path.join(scotch_lib_dir, 'libscotcherr.a')

        # test that we can link a binary using scotch static libs:
        cmdstr = "%s -o %s %s %s %s" % \
                 (linker, app, cpp_file.replace('.cpp', '.o'),
                  libscotch_static, libscotcherr_static)
        linkFailed, cmdoutput = getstatusoutput(cmdstr)
        if not linkFailed:
            break

    if linkFailed:
        remove_cppfile(cpp_file, ofile=True)
        raise UnableToLinkException("SCOTCH", cmd=cmdstr,
                                    program=cpp_test_lib_str,
                                    errormsg=cmdoutput)
   
    # test that we can run the binary:
    runFailed, cmdoutput = getstatusoutput(app)
    remove_cppfile(cpp_file, ofile=True, execfile=True)
    if runFailed or not cmdoutput == "success":
        raise UnableToRunException("SCOTCH", errormsg=cmdoutput)

    return "-L%s -lscotch -lscotcherr" % scotch_lib_dir

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
        libs = pkgLibs(compiler=compiler, linker=linker,
                       cflags=cflags, sconsEnv=sconsEnv)
    else:
        # run pkgLibs as this is the current SCOTCH test
        pkgLibs(compiler=compiler, linker=linker,
                cflags=cflags, sconsEnv=sconsEnv) 
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv)
        
    return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  scotch_version, scotch_libs, scotch_cflags = pkgTests(sconsEnv=sconsEnv)

  # Ready to create a pkg-config file:
  pkg_file_str = r"""Name: SCOTCH
Version: %s
Description: SCOTCH mesh and graph partitioning, http://www.labri.fr/perso/pelegrin/scotch/
Libs: %s
Cflags: %s 
""" % (scotch_version, scotch_libs, scotch_cflags)
  pkg_file = open(os.path.join(directory, "scotch.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found SCOTCH and generated pkg-config file in\n '%s'" % directory

if __name__ == "__main__":
  generatePkgConf(directory=".")
