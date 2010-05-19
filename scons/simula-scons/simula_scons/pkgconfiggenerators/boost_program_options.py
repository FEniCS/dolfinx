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

  # create a simple test program that uses Boost.Program_options:
  cpp_test_lib_str = r"""
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <string>
#include <iostream>
#include <iterator>
using namespace std;

int main(int argc, char* argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options() ("foo", po::value<string>(), "just an option");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("foo")) {
    cout << "success, foo is " << vm["foo"].as<string>();
  } else {
    cout << "failure";
  }

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
    raise UnableToCompileException("Boost.Program_options", cmd=cmdstr,
                                   program=cpp_test_lib_str,
                                   errormsg=cmdoutput)

  # test that we can link a binary using Boost.Program_options:
  lib_dir = os.path.join(boost.getBoostDir(sconsEnv=sconsEnv), 'lib')
  po_lib = "boost_program_options"
  app = os.path.join(os.getcwd(), "a.out")
  cmdstr = "%s  %s -o %s -L%s -l%s" % \
           (linker, cpp_file.replace('.cpp', '.o'), app, lib_dir, po_lib)
  linkFailed, cmdoutput = getstatusoutput(cmdstr)
  if linkFailed:
    # try to append -mt to lib
    po_lib += "-mt"
    cmdstr = "%s %s -o %s -L%s -l%s" % \
             (linker, cpp_file.replace('.cpp', '.o'), app, lib_dir, po_lib)
    linkFailed, cmdoutput = getstatusoutput(cmdstr)
    if linkFailed:
      remove_cppfile(cpp_file, ofile=True)
      raise UnableToLinkException("Boost.Program_options", cmd=cmdstr,
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
  runFailed, cmdoutput = getstatusoutput(app + ' --foo=ok')
  remove_cppfile(cpp_file, ofile=True, execfile=True)
  if runFailed or not "success" in cmdoutput:
    raise UnableToRunException("Boost.Program_options", errormsg=cmdoutput)

  return "-L%s -l%s" % (lib_dir, po_lib)

# Overwrite the pkgLibs function from boost.py.
boost.pkgLibs = pkgLibs

def generatePkgConf(directory=None, sconsEnv=None, **kwargs):

  if directory is None:
    directory = suitablePkgConfDir()

  version, libs, cflags = boost.pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: Boost.Program_options
Version: %s
Description: The Boost program_options library for obtaining program options
Libs: %s
Cflags: %s
""" % (version, repr(libs)[1:-1], repr(cflags)[1:-1])
  # FIXME:      ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
  # Is there a better way to handle this on Windows?
  
  pkg_file = open(os.path.join(directory, "boost_program_options.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found Boost.Program_options and generated pkg-config file in" \
        "\n '%s'" % directory

if __name__ == "__main__":
  generatePkgConf(directory=".")
