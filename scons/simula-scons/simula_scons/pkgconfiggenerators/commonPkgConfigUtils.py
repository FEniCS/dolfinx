#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright C (2006) Simula Research Laboratory
# Author: Åsmund Ødegård
#
# Idea: search the PKG_CONFIG_PATH environment variable for writeable 
# directory. If such a directory is found, return this directory for use 
# by other pkg-config generators.
#
# If no such directory is found, report the user that PKG_CONFIG_PATH must 
# contain a writeable directory.
# 
# An other build system which name should not be mentioned here, but those who 
# care know what I'm talking about, did something like that with me once. All I 
# can tell you is this: It was not nice.

import os
import os.path

class NoWritablePkgConfDir(Exception):
    def __init__(self, directory, msg=""):
        Exception.__init__(self," Unable to use '%s' as PKG_CONFIG directory. (%s)\nPlease, add a writeable directory in your PKG_CONFIG_PATH variable." % (directory,msg))

def suitablePkgConfDir():
    # try:
    #         pkgConfDirs = os.environ["PKG_CONFIG_PATH"].split(os.pathsep)
    #     except:
    #         pkgConfDirs = []
        
    # return first writeable directory in the PKG_CONFIG_PATH environment:
    #
    # Most people just don't want this, so we have to make a slight change...
    #
    # * we don't have a scons env here, and I'm not sure we want one, so grabbing
    # options from the env is probably not what we want here... 
    # * A possible solution could be that we read a --pkg-config-dir option
    # in scons, and update the os.environ with the SCONS_PKG_CONFIG_DIR variable
    # as used below. 
    # for directory in pkgConfDirs:
    #     try:
    #         testfilename = "%s/__testpkgconfwritablefile" % (directory)
    #         tmpfile = open(testfilename,'w')
    #         tmpfile.write("test")
    #         tmpfile.close()
    #         os.unlink(testfilename)
    #         return directory
    #     except:
    #         pass
    
    # if we are unable to find a suitable directory     
    # If we have SCONS_PKG_CONFIG_DIR defined, we use that, else we use
    # the current directory
    
    if os.environ.has_key("SCONS_PKG_CONFIG_DIR"):
      directory = os.environ["SCONS_PKG_CONFIG_DIR"]
      try:
        # The try/except should only be for the write-test
        if not os.path.isdir(directory):
          os.makedirs(directory)
        testfilename = "%s/__testpkgconfwritablefile" % (directory)
        tmpfile = open(testfilename,'w')
        tmpfile.write("test")
        tmpfile.close()
        os.unlink(testfilename)
        # also, put this directory early in PKG_CONFIG_DIR if not there already (which it should)
        if os.environ.has_key("PKG_CONFIG_PATH") and os.environ["PKG_CONFIG_PATH"] != "" \
            and not directory in os.environ["PKG_CONFIG_PATH"]:
          pkgConfPath = os.environ["PKG_CONFIG_PATH"].split(os.path.pathsep)
          pkgConfPath.append(directory)
          os.environ["PKG_CONFIG_PATH"] = os.path.pathsep.join(pkgConfPath)
          print """\n** Warning: consider adding %s
in your PKG_CONFIG_PATH permanently**""" % (directory)

        elif (os.environ.has_key("PKG_CONFIG_PATH") and os.environ["PKG_CONFIG_PATH"] == "") \
             or not os.environ.has_key("PKG_CONFIG_PATH"): 
          os.environ["PKG_CONFIG_PATH"] = directory
          print """\n** Warning: consider adding %s
in your PKG_CONFIG_PATH permanently**""" % (directory)
        return directory
      except:
        pass
 
    # finally, resort to the current directory.
    directory = os.path.dirname(__file__)
    # ensure that we can write in this directory (using someone else' src-tree?)
    try:
        testfilename = "%s/__testpkgconfwritablefile" % (directory)
        tmpfile = open(testfilename,'w')
        tmpfile.write("test")
        tmpfile.close()
        os.unlink(testfilename)
    except:
        raise NoWritablePkgConfDir(directory)
    
    # Add the directory in the PKG_CONFIG_PATH environment, and return it:
    print """\n** Warning **
pkg-config files may be generated in the directory:\n %s.
Consider updating your PKG_CONFIG_PATH variable with this directoy.""" % (directory)
    return directory


def get_architecture():
  #archpipe = os.popen("uname -s")
  #arch = archpipe.readline().rstrip()
  #archpipe.close()

  # Linux: linux*, startswith('linux')
  # MacOSX: darwin
  # Windows: win32, maybe win64? - startswith('win')

  # better:
  import sys
  arch = sys.platform
  return arch

def get_compiler(sconsEnv):
  # if we have a sconsEnv, get CXX compiler from the env, else check os.environ
  # default to g++
  compiler = None
  if sconsEnv:
    compiler = sconsEnv.get("CXX", False)
  if not compiler:
    if os.environ.has_key("CXX"):
      compiler = os.environ["CXX"]
    else:
      compiler = "g++"
  return compiler

def get_alternate_cxx_compiler(compiler, arch=None):
  if not arch:
    arch = get_architecture()

  dir_c = os.path.dirname(compiler)
  base_c = os.path.basename(compiler)

  mpicc_alternates = ["mpicxx", "mpic++", "mpiCC"]
  gcc_alternates = ["g++", "c++", "CC"]
  cc_alternates = ["CC", "c++", "g++"]

  if arch.startswith('win'):
    mpicc_alternates = ["%s.exe" % a for a in mpicc_alternates]
    gcc_alternates = ["%s.exe" % a for a in gcc_alternates]
    cc_alternates = ["%s.exe" % a for a in cc_alternates]

  if base_c == "mpicc":
    use_alternates = mpicc_alternates
  elif base_c == "gcc":
    use_alternates = gcc_alternates
  elif base_c == "cc":
    use_alternates = cc_alternates
  else:
    # Can not find an alternate, stay with the default.
    print "Does not know about any alternates for %s, so that is used" % (base_c)
    return compiler

  if dir_c == "":
    # compiler given as something in $PATH, not fully qualified
    # If arch is windows, test with the .exe extension (is that the only valid 
    # for binaries?)
    # In this case, check for alternate in $PATH, but stay with non-qualified in 
    # the result.
    alternates = [fp for fp in \
          [os.path.join(p,f) for p in os.environ['PATH'].split(os.pathsep) for f in use_alternates ] \
          if os.path.exists(fp)]
    if len(alternates) > 0:
      return os.path.basename(alternates[0])
    else:
      print "Can not find alternate for %s" % base_c
      return base_c
  else:
    # Full path given, search just that path for alternative.
    alternates = [fp for fp in \
        [os.path.join(dir_c,f) for f in use_alternates] \
        if os.path.exists(fp)]
    if len(alternates) > 0:
      # return the basename of the first valid alternative
      # We let PATH on the userside handle things
      return alternates[0]
    else:
      print "Can not find alternate for %s" % base_c
      return compiler

def get_linker(sconsEnv):
  # if we have a sconsEnv, get LD linker from the env, else check os.environ.
  # default to g++
  linker = None
  if sconsEnv:
    linker = sconsEnv.get("LD", False)
  if not linker:
    if os.environ.has_key("LD"):
      linker = os.environ["LD"]
    else:
      linker = "g++"
  return linker

def set_forced_compiler(forceCompiler):
  # If forceCompiler is set (because we want to test with another compiler), 
  # force the first entry in the forceCompiler as compiler, the second as linker
  # If forceCompiler is just a singel string, use as both compiler and linker
  if isinstance(forceCompiler, (tuple,list)):
    compiler = forceCompiler[0]
    linker = forceCompiler[1]
  elif isinstance(forceCompiler, str):
    compiler = forceCompiler
    linker = forceCompiler
  else:
    raise Exception ("Tried to force Compiler/Linker using unknown syntax")
  return compiler, linker

def write_cppfile(code_str, name):
  cpp_file = open(name,'w')
  cpp_file.write(code_str)
  cpp_file.close()

def write_cfile(code_str, name):
    write_cppfile(code_str, name)

def remove_cppfile(name, execfile=False, ofile=False, libfile=False):
  if execfile:
    os.unlink("a.out")
  if ofile:
    basename = ".".join(name.split(".")[:-1])
    os.unlink("%s.o" % (basename))
  if libfile:
    basename = ".".join(name.split(".")[:-1])
    os.unlink("lib%s.so" % (basename))
  os.unlink(name)

def remove_cfile(name, execfile=False, ofile=False, libfile=False):
    remove_cppfile(name, execfile, ofile, libfile)

class UnableToXXXException(Exception):
    def __init__(self, msg="", errormsg=""):
        if errormsg:
            msg += "\nError message:\n%s" % errormsg
        Exception.__init__(self, msg)
        
class UnableToFindPackageException(UnableToXXXException): pass
class UnableToCompileException(UnableToXXXException): pass
class UnableToLinkException(UnableToXXXException): pass
class UnableToRunException(UnableToXXXException): pass
