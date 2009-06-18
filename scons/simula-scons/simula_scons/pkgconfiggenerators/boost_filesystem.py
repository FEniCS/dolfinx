#!/usr/bin/env python

import os
import boost

from commonPkgConfigUtils import *

# Use the same pkgTests function as in boost.py:
pkgTests = boost.pkgTests

def pkgLibs(compiler=None, linker=None, cflags=None, sconsEnv=None):
  if not compiler:
    compiler = get_compiler(sconsEnv)
  if not linker:
    linker = get_linker(sconsEnv)
  if not cflags:
    cflags = boost.pkgCflags(sconsEnv=sconsEnv)

  # create a simple test program that uses Boost.Filesystem:
  cpp_test_lib_str = r"""
#include <boost/filesystem.hpp>
#include <string>

int main(int argc, char* argv[]) {
  const std::string filename = "test.h";
  const boost::filesystem::path path(filename);
  const std::string extension = boost::filesystem::extension(path);

  return 0;
}
"""
  cpp_file = "boost_config_test_lib.cpp"
  write_cppfile(cpp_test_lib_str, cpp_file)

  # test that we can compile:
  cmdstr = "%s %s -c %s" % (compiler, cflags, cpp_file)
  compileFailed, cmdoutput = getstatusoutput(cmdstr)
  if compileFailed:
    remove_cppfile(cpp_file)
    raise UnableToCompileException("Boost.Filesystem", cmd=cmdstr,
                                   program=cpp_test_lib_str,
                                   errormsg=cmdoutput)

  # test that we can link a binary using Boost.Filesystem:
  lib_dir = os.path.join(boost.getBoostDir(sconsEnv=sconsEnv), 'lib')
  filesystem_lib = "boost_filesystem"
  system_lib = "boost_system"
  app = os.path.join(os.getcwd(), "a.out")
  cmdstr = "%s -o %s -L%s -l%s" % (linker, app, lib_dir, filesystem_lib)
  if get_architecture() == "darwin":
    cmdstr += " -l%s" % system_lib
  cmdstr += " %s" % cpp_file.replace('.cpp', '.o')
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # try to append -mt to lib
    filesystem_lib += "-mt"
    system_lib += "-mt"
    cmdstr = "%s -o %s -L%s -l%s" % \
             (linker, app, lib_dir, filesystem_lib)
    if get_architecture() == "darwin":
      cmdstr += " -l%s" % system_lib
    cmdstr += " %s" % cpp_file.replace('.cpp', '.o')
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
      remove_cppfile(cpp_file, ofile=True)
      raise UnableToLinkException("Boost.Filesystem", cmd=cmdstr,
                                  program=cpp_test_lib_str,
                                  errormsg=cmdoutput)
  
  # test that we can run the binary:
  runFailed, cmdoutput = getstatusoutput(app)
  remove_cppfile(cpp_file, ofile=True, execfile=True)
  if runFailed:
    raise UnableToRunException("Boost.Filesystem", errormsg=cmdoutput)

  libs = "-L%s -l%s" % (lib_dir, filesystem_lib)
  if get_architecture() == "darwin":
    libs += " -l%s" % system_lib
  return libs

# Overwrite the pkgLibs function from boost.py:
boost.pkgLibs = pkgLibs

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):

  if directory is None:
    directory = suitablePkgConfDir()

  version, libs, cflags = boost.pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: Boost.Filesystem
Version: %s
Description: The Boost filesystem library for manipulating paths, files, and directories
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
  # FIXME:      ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
  # Is there a better way to handle this on Windows?
  
  pkg_file = open(os.path.join(directory, "boost_filesystem.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found Boost.Filesystem and generated pkg-config file in" \
        "\n '%s'" % directory

if __name__ == "__main__":
  generatePkgConf(directory=".")
