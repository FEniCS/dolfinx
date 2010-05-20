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
  libs = [filesystem_lib]
  system_lib = "boost_system"
  app = os.path.join(os.getcwd(), "a.out")
  cmdstr = "%s %s -o %s -L%s -l%s" % (linker, cpp_file.replace('.cpp', '.o'),
                                      app, lib_dir, filesystem_lib)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # on newer versions of Boost, Boost.FileSystem requires to link
    # with Boost.System:
    libs.append(system_lib)
    cmdstr = "%s %s -o %s -L%s -l%s -l%s" % \
             (linker, cpp_file.replace('.cpp', '.o'),
              app, lib_dir, filesystem_lib, system_lib)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # try to append -mt to lib
    filesystem_lib += "-mt"
    libs = [filesystem_lib]
    cmdstr = "%s %s -o %s -L%s -l%s" % (linker, cpp_file.replace('.cpp', '.o'),
                                        app, lib_dir, filesystem_lib)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # again, try to link with Boost.System:
    system_lib += "-mt"
    libs.append(system_lib)
    cmdstr = "%s %s -o %s -L%s -l%s -l%s" % \
             (linker, cpp_file.replace('.cpp', '.o'),
              app, lib_dir, filesystem_lib, system_lib)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    remove_cppfile(cpp_file, ofile=True)
    raise UnableToLinkException("Boost.Filesystem", cmd=cmdstr,
                                program=cpp_test_lib_str,
                                errormsg=cmdoutput)
  
  # test that we can run the binary:
  arch = get_architecture()
  if arch == 'darwin':
    os.putenv('DYLD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('DYLD_LIBRARY_PATH', ''), lib_dir]))
  elif arch.startswith('win'):
    os.putenv('PATH', os.pathsep.join([os.getenv('PATH', ''), lib_dir]))
  else:
    os.putenv('LD_LIBRARY_PATH',
              os.pathsep.join([os.getenv('LD_LIBRARY_PATH', ''), lib_dir]))
  runFailed, cmdoutput = getstatusoutput(app)
  remove_cppfile(cpp_file, ofile=True, execfile=True)
  if runFailed:
    raise UnableToRunException("Boost.Filesystem", errormsg=cmdoutput)

  return "-L%s %s" % (lib_dir, ' '.join(["-l%s" % l for l in libs]))

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
