#!/usr/bin/env python
import os,sys
import string
import os.path

from commonPkgConfigUtils import *

def getTrilinosDir(sconsEnv=None):
    if get_architecture() == "darwin":
        # use fink as default
        default = "/sw"
    else:
        default = "/usr"
    trilinos_dir = getPackageDir("trilinos", sconsEnv=sconsEnv, default=default)
    return trilinos_dir

def pkgVersion(sconsEnv=None):
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
    package_locations = ["site-packages", "dist-packages"]
    pytrilinos_dir = []
    for location in package_locations:
       pytrilinos_dir.append("%s/lib/python%s/%s" % \
                     (getTrilinosDir(sconsEnv=sconsEnv), pyversion, location))

    if len(set(pytrilinos_dir).intersection(set(sys.path))) ==  0:
        print """The Python site-packages directory for Trilinos is not 
in your PYTHONPATH, consider adding one of
%s 
to PYTHONPATH in your environment. You will probably need to adjust
LD_LIBRARY_PATH/DYLD_LIBRARY_PATH as well.
""" % pytrilinos_dir

    trilinos_version = 6
    try:
        import PyTrilinos
        trilinos_version = PyTrilinos.version().split()[2]
    except:
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

            if epetra_version >= 3.6 and ml_version >= 6.1 and \
                   aztecoo_version >= 3.6:
                trilinos_version = 8
            elif epetra_version >= 3.5 and ml_version >= 5.0 and \
                     aztecoo_version >= 3.5:
                trilinos_version = 7
            else:
                trilinos_version = 6
        except:
            # Unable to load PyTrilinos, assume v.6 of trilinos
            pass
    
    return trilinos_version

def pkgCflags(sconsEnv=None):
    trilinos_dir = getTrilinosDir(sconsEnv=sconsEnv)
    trilinos_inc_dir = None
    inc_dirs = ['include', os.path.join('include', 'trilinos')]
    for inc_dir in inc_dirs:
        if os.path.exists(os.path.join(trilinos_dir, inc_dir, 'ml_config.h')):
            trilinos_inc_dir = os.path.join(trilinos_dir, inc_dir)
            break
    
    if trilinos_inc_dir is None:
        raise UnableToFindPackageException("Trilinos")
    
    return "-I%s" % trilinos_inc_dir

def pkgLibs(compiler=None, cflags=None, sconsEnv=None):
    cpp_file_str = r""" 
#include <ml_config.h>
#include <stdio.h>

int main() {

#ifdef HAVE_ML_TRIUTILS
  printf("-ltriutils\n");
#endif

printf("-lml\n");

#ifdef HAVE_ML_NOX
  printf("-lnox\n");'
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
        compiler = get_compiler(sconsEnv)
    if not cflags:
        cflags = pkgCflags(sconsEnv=sconsEnv)

    cmdstr = "%s -o a.out %s trilinos_config_test.cpp" % (compiler,cflags)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile("trilinos_config_test.cpp")
        raise UnableToCompileException("Trilinos", cmd=cmdstr,
                                       program=cpp_file_str, errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    if runFailed:
        remove_cppfile("trilinos_config_test.cpp", execfile=True)
        raise UnableToRunException("Trilinos", errormsg=cmdoutput)

    libs_dir = os.path.join(getTrilinosDir(sconsEnv=sconsEnv), "lib")
    libs_str = " -L%s %s" % (libs_dir,string.join(string.split(cmdoutput, '\n')))
    
    # Check if we should prefix the Trilinos libraries with trilinos_ 
    # as in the latest debian packages:
    if not (os.path.exists(os.path.join(libs_dir, "libml.a")) or \
            os.path.exists(os.path.join(libs_dir, "libml.so"))) and \
           (os.path.exists(os.path.join(libs_dir, "libtrilinos_ml.a")) or \
            os.path.exists(os.path.join(libs_dir, "libtrilinos_ml.so"))):
        libs_str = libs_str.replace('-l', '-ltrilinos_')
    
    remove_cppfile("trilinos_config_test.cpp", execfile=True)

    # We also have to link with BLAS and LAPACK:
    arch = get_architecture()
    blasstring = ""
    if arch == "darwin":
    	libs_str += " -framework vecLib"
    else:
        libs_str += " -L%s -lblas -llapack" % getAtlasDir(sconsEnv=sconsEnv)

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
        cflags = pkgCflags(sconsEnv=sconsEnv)
    if not libs:
        libs = pkgLibs(compiler=compiler, cflags=cflags, sconsEnv=sconsEnv)
    if not version:
        version = pkgVersion(sconsEnv=sconsEnv)

    # Check that Trilinos has been built with MPI support
    cpp_file_str = r""" 
#include <ml_config.h>
#include <stdio.h>

int main() {
#ifdef HAVE_MPI
  printf("yes");
#endif
}
"""
    cpp_file = "trilinos_test_mpi.cpp"
    write_cppfile(cpp_file_str, cpp_file)

    cmdstr = "%s -o a.out %s %s" % (compiler, cflags, cpp_file)
    compileFailed, cmdoutput = getstatusoutput(cmdstr)
    if compileFailed:
        remove_cppfile(cpp_file)
        raise UnableToCompileException("Trilinos", cmd=cmdstr,
                                       program=cpp_file_str, errormsg=cmdoutput)

    cmdstr = os.path.join(os.getcwd(), "a.out")
    runFailed, cmdoutput = getstatusoutput(cmdstr)
    remove_cppfile(cpp_file, execfile=True)
    if runFailed:
        raise UnableToRunException("Trilinos", errormsg=cmdoutput)
    if not "yes" in cmdoutput:
        msg = "Trilinos must be built with MPI support."
        raise UnableToRunException("Trilinos", errormsg=msg)

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


