#!/usr/bin/env python
import os,sys
import string
import os.path
import commands

from commonPkgConfigUtils import *

def getTrilinosDir(**kwargs):
    trilinos_dir = getPackageDir("trilinos",
                                 sconsEnv=kwargs.get("sconsEnv", None),
                                 default=os.path.join(os.path.sep,"usr","local"))
    return trilinos_dir

def pkgVersion(**kwargs):
    # Try to figure out what version of Trilinos we have.
    # Some of these imports try to load NumPy, which is also present 
    # in the current directory. Hence we need to remove the current 
    # directory from sys.path:

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
            del sys.path[sys.path.index(file_abspath)]
        except:
            # not in sys.path anymore!
            remove_path = False
    while remove_realpath:
      try:
        del sys.path[sys.path.index(realfile_abspath)]
      except:
        # not in sys.path anymore!
        remove_realpath = False
    # Mayby this works too:
    #while file_abspath in sys.path:
    #    del sys.path[sys.path.index(file_abspath)]
    #while realfile_abspath in sys.path:
    #    del sys.path[sys.path.index(realfile_abspath)]

    # If TRILINOS_DIR/lib/python{version}/site_packages is not in PYTHONPATH
    # the loading of PyTrilinos will fail. Add it and message user.
    pyversion = ".".join([str(s) for s in sys.version_info[0:2]])
    pytrilinos_dir = "%s/lib/python%s/site-packages" % \
                     (getTrilinosDir(**kwargs),pyversion)
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
            raise UnableToXXXException("Unable to figure out the version of ML")
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
    
    return trilinos_version

def pkgCflags(**kwargs):
    include_dir = os.path.join(getTrilinosDir(**kwargs), "include")
    cflags = "-I%s" % include_dir
    return cflags

def pkgLibs(compiler=None, cflags=None, **kwargs):
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

    if not compiler:
        compiler = get_compiler(kwargs.get("sconsEnv", None))
    if not cflags:
        cflags = pkgCflags(**kwargs)

    cmdstr = "%s %s trilinos_config_test.cpp" % (compiler,cflags)
    compileFailed, cmdoutput = commands.getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile("trilinos_config_test.cpp")
        raise UnableToCompileException("Trilinos", cmd=cmdstr,
                                       program=cpp_file_str, errormsg=cmdoutput)

    runFailed, cmdoutput = commands.getstatusoutput("./a.out")
    if runFailed:
        remove_cppfile("trilinos_config_test.cpp", execfile=True)
        raise UnableToRunException("Trilinos", errormsg=cmdoutput)

    libs_dir = os.path.join(getTrilinosDir(**kwargs), "lib")
    libs_str = " -L%s %s" % (libs_dir,string.join(string.split(cmdoutput, '\n')))
    
    remove_cppfile("trilinos_config_test.cpp", execfile=True)

    # We also have to link with BLAS and LAPACK:
    arch = get_architecture()
    blasstring = ""
    if arch == "darwin":
    	libs_str += " -framework vecLib"
    else:
        libs_str += " -L%s -lblas -llapack" % getAtlasDir(**kwargs)

    return libs_str

def pkgTests(forceCompiler=None, sconsEnv=None,
             cflags=None, libs=None, version=None, **kwargs):
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

    if not forceCompiler:
        compiler = get_compiler(sconsEnv)
        linker = get_linker(sconsEnv)
    else:
        compiler, linker = set_forced_compiler(forceCompiler)

    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv, **kwargs)
    if not libs:
        libs = pkgLibs(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)
    else:
        # Force a call to pkgLibs since this is the test for this package:
        libs = pkgLibs(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)
    if not version:
        version = pkgVersion(**kwargs)

    return version, cflags, libs

def generatePkgConf(directory=suitablePkgConfDir(), sconsEnv=None, **kwargs):
    
    trilinos_version, trilinos_cflags, trilinos_libs = pkgTests(sconsEnv=sconsEnv)

    pkg_file_str = r"""Name: Trilinos
Version: %s
Description: The Trilinos project - http://software.sandia.gov/trilinos
Libs: %s
Cflags: %s
""" % (trilinos_version, trilinos_libs, trilinos_cflags)

    pkg_file = open("%s/trilinos.pc" % directory, 'w')
    pkg_file.write(pkg_file_str)
    pkg_file.close()

    print "done\n Found Trilinos and generated pkg-config file in \n '%s'" % directory

if __name__ == "__main__":
    generatePkgConf(directory=".")


