#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

def getBoostDir():
  # Try to get boost_dir from environment variable, if not use some default.
  arch = get_architecture()
  if os.environ.has_key("BOOST_DIR"):
    boost_dir = os.environ["BOOST_DIR"]
  else:
    if arch == "darwin":
      # use fink as default
      boost_dir = os.path.join(os.path.sep,"sw")
    else:
      boost_dir = os.path.join(os.path.sep,"usr")
  return boost_dir

def pkgVersion(compiler=None, cflags=None, **kwargs):
  """Find the Boost version."""
  # This is a bit special. It is given in the library as
  # a 6 digit number, like 103301. We have to do some arithmetics
  # to find the real version:
  # VERSION / 100000 => major version (1 in this case)
  # VERSION / 100 % 1000 => minor version (33 in this case)
  # VERSION % 100 => sub-minor version (1 in this case).
  # 
  # The version check also verify that we can include some boost headers.
  cpp_version_str = r"""
#include <boost/version.hpp>
#include <iostream>
int main() {
#ifdef BOOST_VERSION
  std::cout << BOOST_VERSION;
#endif
return 0;
}
"""
  write_cppfile(cpp_version_str, "boost_config_test_version.cpp")

  if not compiler:
    sconsEnv = kwargs.get('sconsEnv', None)
    compiler = get_compiler(sconsEnv)
  if not cflags:
    cflags = pkgCflags()
  cmdstr = "%s %s boost_config_test_version.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    msg = ("Unable to compile some Boost tests, using '%s' as " + \
           "location for Boost includes.") % getBoostDir()
    raise UnableToCompileException(msg, cmdoutput)

  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed:
    raise UnableToRunException(errormsg=cmdoutput)
  boost_version = int(cmdoutput)
  boost_major = boost_version/100000
  boost_minor = boost_version / 100 % 1000
  boost_subminor = boost_version % 100
  full_boost_version = "%s.%s.%s" % (boost_major, boost_minor, boost_subminor)

  remove_cppfile("boost_config_test_version.cpp", execfile=True)
  
  return full_boost_version

def pkgLibs(**kwargs):
  return ""

def pkgCflags(**kwargs):
  include_dir = os.path.join(getBoostDir(),"include")
  return "-I%s" % include_dir

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
  """Run the tests for this package.
     
     If Ok, return various variables, if not we will end with an exception.
     forceCompiler, if set, should be a tuple containing (compiler, linker)
     or just a string, which in that case will be used as both
  """
  # set which compiler and linker to use:
  if not forceCompiler:
    compiler = get_compiler(sconsEnv)
    linker = get_linker(sconsEnv)
  else:
    compiler, linker = set_forced_compiler(forceCompiler)

  if not version:
    version = pkgVersion(compiler=compiler)
  if not cflags:
    cflags = pkgCflags()
  if not libs:
    libs = pkgLibs()
  
  # All we want to do is to compile in some boost headers, so really know
  # enough already, as the API of the headers are defined by the version.
  cpp_testublas_str = r"""
#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
int main() {
  boost::numeric::ublas::vector<double> ubv(10);
  if ( ubv.size() == 10 ) {
    std::cout << "ublas ok";
  } else {
    std::cout << "ublas not ok";
  }
}
"""
  write_cppfile(cpp_testublas_str, "boost_config_test_ublas.cpp")

  cmdstr = "%s %s boost_config_test_ublas.cpp" % (compiler, cflags)
  compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
  if compileFailed:
    msg = ("Unable to compile some Boost tests, using '%s' as " + \
          "location for Boost includes.") % getBoostDir()
    raise UnableToCompileException(msg, cmdoutput)

  runFailed, cmdoutput = commands.getstatusoutput("./a.out")
  if runFailed or cmdoutput != "ublas ok":
    raise UnableToRunException(errormsg=cmdoutput)

  remove_cppfile("boost_config_test_ublas.cpp", execfile=True)

  return version, libs, cflags

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):

  version, libs, cflags = pkgTests(sconsEnv=sconsEnv)

  pkg_file_str = r"""Name: Boost
Version: %s
Description: The Boost library of template code
Libs: %s
Cflags: %s
""" % (version, libs, cflags)
  
  pkg_file = open(os.path.join(directory,"boost.pc"), 'w')
  pkg_file.write(pkg_file_str)
  pkg_file.close()
  print "done\n Found Boost and generated pkg-config file in\n '%s'" % \
        directory

if __name__ == "__main__":
  generatePkgConf(directory=".")
