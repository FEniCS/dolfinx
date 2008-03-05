#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *


class unableToCompileException(Exception):
    def __init__(self, trilinos_dir="", errormsg=""):
        Exception.__init__(self, "Unable to compile some Trilinos tests, using %s as location for Trilinos includes.\nAdditional message:\n%s" % (trilinos_dir, errormsg))

class unableToLinkException(Exception):
    def __init__(self, trilinos_dir="", errormsg=""):
        Exception.__init__(self, "Unable to link some Trilinos tests, using %s as location for Trilinos libs.\nAdditional message:\n%s" % (trilinos_dir, errormsg))

class unableToRunException(Exception):
    def __init__(self):
        Exception.__init__(self, "Unable to run Trilinos program correctly.")

def pkgTests(forceCompiler=None, sconsEnv=None, **kwargs):
    """Run the tests for this package
     
    If Ok, return various variables, if not we will end with an exception.
    forceCompiler, if set, should be a tuple containing (compiler, linker)
    or just a string, which in that case will be used as both
    """
    
    # play tricks with pythonpath to avoid mixing up 
    # the real numpy with the one in this directory.
    file_abspath = os.path.abspath(os.path.dirname(__file__))
    restore_environ = False
    try:
        splitpythonpath = os.environ['PYTHONPATH'].split(os.pathsep)
        index_filepath = splitpythonpath.index(file_abspath)
        del(splitpythonpath[index_filepath])
        os.environ['PYTHONPATH'] = os.path.pathsep.join(splitpythonpath)
        restore_environ = True
    except:
        pass
 
    cpp_file_str = r""" 
#include <ml_config.h>
#include <stdio.h>

int main() {

#ifdef HAVE_ML_TRIUTILS
  printf("-ltriutils\n");
#endif

printf("-lml\n");

#ifdef HAVE_ML_NOX
  printf("-lnox\n")'
#endif

#ifdef HAVE_ML_IFPACK
  printf("-lifpack\n");
#endif

#ifdef HAVE_ML_AMESOS 
  printf("-lamesos\n"); 
#endif 

#ifdef HAVE_ML_AZTECOO 
  printf("-laztecoo\n"); 
#endif 

#ifdef HAVE_ML_ANASAZI 
  printf("-lanasazi\n"); 
#endif

#ifdef HAVE_ML_TEUCHOS
  printf("-lteuchos\n");
#endif 

#ifdef HAVE_ML_EPETRA 
  printf("-lepetra\n"); 
#endif 

#ifdef HAVE_ML_EPETRAEXT 
  printf("-lepetraext\n"); 
#endif 

#ifdef HAVE_ML_GALERI
  printf("-lgaleri\n");
#endif
}
"""

    write_cppfile(cpp_file_str, "trilinos_config_test.cpp")

    # If a SCons environment is supplied, and it has a withTrilinosDir setting:
    # If not, try the TRILINOS_DIR environment variable
    # Else try /usr/local
    trilinos_dir = ""
    if os.environ.has_key('TRILINOS_DIR'):
        trilinos_dir=os.environ["TRILINOS_DIR"]
    else:
        trilinos_dir=os.path.join(os.path.sep, "usr","local")

    # Set ATLAS location from environ, or use default
    atlaslocation = ""
    if os.environ.has_key('ATLAS_DIR'):
        atlaslocation = os.environ["ATLAS_DIR"]
    else:
    	atlaslocation = os.path.join(os.path.sep, "usr","lib","atlas")

    if not forceCompiler:
        compiler = get_compiler(sconsEnv)
        linker = get_linker(sconsEnv)
    else:
        compiler, linker = set_forced_compiler(forceCompiler)

    cmdstr = "%s -I%s/include -c trilinos_config_test.cpp" % (compiler,trilinos_dir)
    compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if compileFailed:
        raise unableToCompileException(trilinos_dir=trilinos_dir, errormsg=cmdoutput)
    cmdstr = "%s trilinos_config_test.o" % linker
    compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if compileFailed:
        raise unableToLinkException(trilinos_dir=trilinos_dir, errormsg=cmdoutput)
    runFailed, cmdoutput = commands.getstatusoutput("./a.out")
    if runFailed:
        raise unableToRunException()
    libs_str = string.join(string.split(cmdoutput, '\n'))
    
    remove_cppfile("trilinos_config_test.cpp", ofile=True, execfile=True)

    arch = get_architecture()
    blasstring = ""
    if arch == "darwin":
    	blasstring = "-framework vecLib"
    else:
    	blasstring = "-L%s -lblas -llapack" % (atlaslocation)
	

    # Next, we try to figure out what version of trilinos we have.
    # Some of these imports try to load numpy, which is also present 
    # in the current directory. Hence we need to remove the current 
    # dir. from sys.path:

    # there may or may not be a symlink involved, remove both 
    # with and without os.path.realpath
    file_abspath = os.path.abspath(os.path.dirname(__file__))
    realfile_abspath = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))

    # may appear several times in sys.path, so remove as long as there 
    # are more:
    remove_path = True
    remove_realpath = True
    while remove_path:
        try:
            del(sys.path[sys.path.index(file_abspath)])
        except:
            # not in sys.path anymore!
            remove_path = False
    while remove_realpath:
      try:
        del(sys.path[sys.path.index(realfile_abspath)])
      except:
        # not in sys.path anymore!
        remove_realpath = False
    # Mayby this works too:
    #while file_abspath in sys.path:
    #    del(sys.path[sys.path.index(file_abspath)])
    #while realfile_abspath in sys.path:
    #    del(sys.path[sys.path.index(realfile_abspath)])

    # If TRILINOS_DIR/lib/python{version}/site_packages is not in PYTHONPATH
    # the loading of PyTrilinos will fail. Add it and message user.
    pyversion=".".join([str(s) for s in sys.version_info[0:2]])
    pytrilinos_dir = "%s/lib/python%s/site-packages" % (trilinos_dir,pyversion)
    try: 
        sys.path.index(pytrilinos_dir)
    except:
        sys.path.append(pytrilinos_dir)
        print """The Python site-packages directory for Trilinos is not 
in your PYTHONPATH, consider adding 
%s 
to PYTHONPATH in your environment. You will probably need to adjust
LD_LIBRARY_PATH/DYLD_LIBRARY_PATH as well.
""" % pytrilinos_dir
   
    trilinos_version = 6
    try:
      import PyTrilinos.Epetra
      import PyTrilinos.ML
      import PyTrilinos.AztecOO

      epetra_version = float(PyTrilinos.Epetra.__version__)
      if hasattr(PyTrilinos.ML, 'VERSION'):
        ml_version = float(PyTrilinos.ML.VERSION)
      elif hasattr(PyTrilinos.ML, 'PACKAGE_VERSION'):
        ml_version = float(PyTrilinos.ML.PACKAGE_VERSION)
      else:
        raise Exception("Unable to figure out the version of ML")
      aztecoo_version = float(PyTrilinos.AztecOO.AztecOO_Version().split()[2])

      # must be expanded for later versions of trilinos
      # TODO: It is also possible to use PyTrilinos.version() to check Trilinos
      # version and PyTrilinos version (at least in version 8)
      #trilinos_version = PyTrilinos.version().split()[2]
      #pytrilinos_version = PyTrilinos.version().split()[5]
      if epetra_version >= 3.6 and ml_version >= 6.1 and aztecoo_version >= 3.6:
          trilinos_version = 8
      elif epetra_version >= 3.5 and ml_version >= 5.0 and aztecoo_version >= 3.5:
          trilinos_version = 7
      else:
          trilinos_version = 6
    except:
        # Unable to load PyTrilinos, assume v.6 of trilinos
        pass

    return trilinos_version, trilinos_dir, libs_str, blasstring

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    
    trilinos_version, trilinos_dir, libs_str, blasstring = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: Trilinos
Version: %s.0
Description: The Trilinos project - http://software.sandia.gov/trilinos
Libs: -L%s/lib %s %s
Cflags: -I%s/include 
""" % (trilinos_version, trilinos_dir, libs_str, blasstring, trilinos_dir)

    pkg_file = open("%s/trilinos-%s.pc" % (directory,trilinos_version), 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()

    print "done\n Found '%s' and generated pkg-config file in \n %s" % ("Trilinos", directory)

if __name__ == "__main__":
    generatePkgConf(directory=".")


