# -*- coding: utf-8 -*-

import os, os.path, sys

# Make sure that we have a good scons-version
EnsureSConsVersion(0, 96)

# Import the local 'scons'
try:
  import simula_scons as scons
except ImportError:
  # Add simula-scons to sys.path and PYTHONPATH
  os.environ["PYTHONPATH"] = \
      os.pathsep.join([os.environ.get("PYTHONPATH",""),
                       os.path.join(os.getcwd(),"scons","simula-scons")])
  sys.path.insert(0, os.path.join(os.getcwd(),"scons","simula-scons"))
  import simula_scons as scons
 
# Import local exceptions
from simula_scons.Errors import PkgconfigError, PkgconfigMissing

# Create a SCons Environment based on the main os environment
env = Environment(ENV=os.environ)

# Set a projectname. Used in some places, like pkg-config generator
env["projectname"] = "dolfin"

scons.setDefaultEnv(env)

# Specify a file where SCons store file signatures, used to figure out what is
# changed. We store it in the 'scons' subdirectory to avoid mixing signatures
# with the files in dolfin. 
dolfin_sconsignfile = os.path.join(os.getcwd(), "scons", ".sconsign")
env.SConsignFile(dolfin_sconsignfile)

# -----------------------------------------------------------------------------
# Command line option handling
# -----------------------------------------------------------------------------

DefaultPackages = ""

# Build the commandline options for SCons:
options = [
    # configurable options for installation:
    scons.PathOption("prefix", "Installation prefix", "/usr/local"),
    scons.PathOption("binDir", "Binary installation directory", "$prefix/bin"),
    scons.PathOption("libDir", "Library installation directory", "$prefix/lib"),
    scons.PathOption("pkgConfDir", "Directory for installation of pkg-config files", "$prefix/lib/pkgconfig"),
    scons.PathOption("includeDir", "C/C++ header installation directory", "$prefix/include"),
    scons.PathOption("pythonModuleDir", "Python module installation directory", 
                     scons.defaultPythonLib(prefix="$prefix")),
    scons.PathOption("pythonExtDir", "Python extension module installation directory", 
                     scons.defaultPythonLib(prefix="$prefix", plat_specific=True)),
    # configurable options for how we want to build:
    BoolOption("enableDebug", "Build with debug information", 1),
    BoolOption("enableDebugUblas", "Add some extra Ublas debug information", 0),
    BoolOption("enableOptimize", "Compile with optimization", 0),
    BoolOption("enableDocs", "Build documentation", 0),
    BoolOption("enableDemos", "Build demos", 0),
    BoolOption("enableProjectionLibrary", "Enable projection library", 0),
    # Enable or disable external packages.
    # These will also be listed in scons.cfg files, but if they are 
    # disabled here, that will override scons.cfg. Remark that unless the
    # module is listed as OptDependencies in scons.cfg, the whole module
    # will be turned off.
    BoolOption("enableMpi", "Compile with support for MPI", "yes"),
    BoolOption("enablePetsc", "Compile with support for PETSc linear algebra", "yes"),
    BoolOption("enableSlepc", "Compile with support for SLEPc", "yes"),
    BoolOption("enableScotch", "Compile with support for SCOTCH graph partitioning", "yes"),
    BoolOption("enableGts", "Compile with support for GTS", "yes"),
    BoolOption("enableUmfpack", "Compile with support for UMFPACK", "yes"),
    BoolOption("enablePydolfin", "Compile the python wrappers of Dolfin", "yes"),
    # some of the above may need extra options (like petscDir), should we
    # try to get that from pkg-config?
    # It may be neccessary to specify the installation path to the above packages.
    # One can either use the options below (with<Package>Dir) or define the
    # <PACKAGE>_DIR environment variable.
    PathOption("withPetscDir", "Specify path to PETSc", None),
    PathOption("withSlepcDir", "Specify path to SLEPc", None),
    PathOption("withScotchDir", "Specify path to SCOTCH", None),
    PathOption("withUmfpackDir", "Specify path to UMFPACK", None),
    PathOption("withBoostDir", "Specify path to Boost", None),
    #
    # a few more options originally from PyCC:
    #BoolOption("autoFetch", "Automatically fetch datafiles from (password protected) SSH repository", 0),
    BoolOption("cacheOptions", "Cache command-line options for later invocations", 1),
    BoolOption("veryClean", "Remove the sconsign file during clean, must be set during regular build", 0),
    # maybe we should do this more cleverly. The default in dolfin now is
    # to use mpicxx if that is available...:
    #("CXX", "Set C++ compiler", scons.defaultCxxCompiler()),
    #("FORTRAN", "Set FORTRAN compiler",scons.defaultFortranCompiler()),
    ("customCxxFlags", "Customize compilation of C++ code", ""),
    #("data", "Parameter to the 'fetch' target: comma-delimited list of directories/files to fetch, \
    #        relative to the `data' directory. An empty value means that everything will be fetched.", ""),
    #("sshUser", "Specify the user for the SSH connection used to retrieve data files", ""),
    #('usePackages','Override or add dependency packages, separate with comma', ""),
    #('customDefaultPackages','Override the default set of packages (%r), separate package names with commas' % (DefaultPackages,)),
    ("SSLOG", "Set Simula scons log file", os.path.join(os.getcwd(),"scons","simula_scons.log")),
    ]


# This Configure class handles both command-line options (which are merged into 
# the environment) and autoconf-style tests. A special feature is that it
# remembers configuration parameters (options and results from tests) between
# invocations so that they are re-used when cleaning after a previous build.
configure = scons.Configure(env, ARGUMENTS, options)

# Open log file for writing
scons.logOpen(env)
scons.log("=================== %s log ===================" % env["projectname"])
scons.logDate()

# Writing the simula_scons used to the log:
scons.log("Using simula_scons from: %s" % scons.__file__)

# Notify the user about that options from scons/options.cache are being used:
if not env.GetOption("clean"):
  try:
    lines = file(os.path.join('scons','options.cache')).readlines()
    if lines:
      print "Using options from scons/options.cache"
  except:
    pass

# If we are in very-clean mode, remove the sconsign file. 
if env.GetOption("clean"):
  if env["veryClean"]:
    os.unlink("%s.dblite" % (dolfin_sconsignfile))
    # FIXME: should we also remove the file scons/options.cache?
    
# Default CXX and FORTRAN flags
env["CXXFLAGS"] = "-Wall -pipe -ansi" # -Werror"
#env["SHFORTRANFLAGS"] = "-Wall -pipe -fPIC"

# If Debug is enabled, add -g:
if env["enableDebug"]:
  env.Append(CXXFLAGS=" -DDEBUG -g -Werror")

if not env["enableDebugUblas"]:
  env.Append(CXXFLAGS=" -DNDEBUG")

# if Optimization is requested, use -O3
if env["enableOptimize"]:
  env.Append(CXXFLAGS=" -O3")
else:
  # FIXME: why are we optimizing when enableOptimize is False?
  env.Append(CXXFLAGS=" -O2")

# Set ENABLE_PROJECTION_LIBRARY if enabled
if env["enableProjectionLibrary"]:
  env.Append(CXXFLAGS=" -DENABLE_PROJECTION_LIBRARY")

# Not sure we need this - but lets leave it for completeness sake - if people
# use if for PyCC, and know that dolfin use the same system, they will expect
# it to be here. We should probably discuss whether that is a good argument or
# not. 
# Append whatever custom flags given
if env["customCxxFlags"]:
  env.Append(CXXFLAGS=" " + env["customCxxFlags"])

# Determine which compiler to be used:
cxx_compilers = ["mpic++", "mpicxx", "mpiCC", "c++", "g++", "CC"]
# If CXX is defined in os.environ, we add this first
if os.environ.has_key("CXX"):
  cxx_compilers.insert(0, os.environ["CXX"])
env["CXX"] = env.Detect(cxx_compilers)

# Set MPI compiler and add neccessary MPI flags if enableMpi is True:
if env["enableMpi"]:
  mpi_cxx_compilers = ["mpic++", "mpicxx", "mpiCC"]
  if not env.Detect("mpirun") or not env["CXX"] in mpi_cxx_compilers:
    print "MPI not found (might not work if PETSc uses MPI)."
  else:
    # Found MPI, so set HAS_MPI and IGNORE_CXX_SEEK (mpich2 bug)
    env.Append(CXXFLAGS=" -DHAS_MPI=1 -DMPICH_IGNORE_CXX_SEEK")

if not env["CXX"]:
  print "Unable to find any valid C++ compiler."
  # try to use g++ as default:
  env["CXX"] = "g++"

# process list of packages to be included in allowed Dependencies.
# Do we need this any more? I think we rather pick up (external) packages from
# the scons.cfg files. Actually, I doubt usePackages is ever used?
#env["usePackages"] = scons.resolvePackages(env["usePackages"].split(','),\
#        env.get("customDefaultPackages", DefaultPackages).split(","))

# Figure out if we should fetch datafiles, and set an option in env. 
#doFetch = "fetch" in COMMAND_LINE_TARGETS or (env["autoFetch"]) and not env.GetOption("clean")
#env["doFetch"] = doFetch

# -----------------------------------------------------------------------------
# Call the main SConscript in the project directory
# -----------------------------------------------------------------------------
try:
  # Invoke the SConscript as if it was situated in the build directory, this
  # tells SCons to build beneath this
  buildDataHash = env.SConscript(os.path.join(env["projectname"], "SConscript"), exports=["env", "configure"])
except PkgconfigError, err:
  sys.stderr.write("%s\n" % err)
  Exit(1)

# -----------------------------------------------------------------------------
# Set up build targets
# -----------------------------------------------------------------------------

# default build-targets: shared libs, extension modules, programs, demos, and
# documentation.
for n in buildDataHash["shlibs"] + buildDataHash["extModules"] + \
        buildDataHash["progs"] + buildDataHash["demos"], buildDataHash["docs"]:
  env.Default(n)

# -----------------------------------------------------------------------------
# Set up installation targets
# -----------------------------------------------------------------------------

# install dolfin-convert into binDir:
env.Install(env["binDir"], os.path.join("misc","utils","convert","dolfin-convert"))

# shared libraries goes into our libDir:
for l in buildDataHash["shlibs"]:
  env.Install(env["libDir"], l)

# install header files in the same structure as in the source tree, within
# includeDir/dolfin:
for h in buildDataHash["headers"]:
  # Get the module path relative to the "src" directory
  dpath = os.path.dirname(h.srcnode().path).split(os.path.sep, 1)[1]
  env.Install(os.path.join(env["includeDir"], "dolfin", dpath), h)
# Also, we want the special 'dolfin.h' file to be installed directly in the
# toplevel includeDir. 
if buildDataHash.has_key("dolfin_header") and buildDataHash["dolfin_header"] != "":
  env.Install(env["includeDir"], buildDataHash["dolfin_header"])

## install python scripts in the bin directory
#for s in buildDataHash["pythonScripts"]:
#  env.Install(env["binDir"], s)

if env["enablePydolfin"]:
    # install python modules, usually in site-packages/dolfin
    for m in buildDataHash["pythonModules"]:
        env.Install(os.path.join(env["pythonModuleDir"], "dolfin"), m)

if env["enablePydolfin"]:
    # install extension modules, usually in site-packages
    for e in buildDataHash["extModules"]:
        env.Install(os.path.join(env["pythonExtDir"], "dolfin"), e)

# install generated pkg-config files in $prefix/lib/pkgconfig or other
# specified place
for p in buildDataHash["pkgconfig"]:
  env.Install(env["pkgConfDir"], p)

# grab prefix from the environment, substitute all scons construction variables
# (those prefixed with $...), and create a normalized path:
prefix = os.path.normpath(env.subst(env["prefix"]))
# add '/' (or similar) at the end of prefix if missing:
if not prefix[-1] == os.path.sep:
  prefix += os.path.sep

# not sure we need common.py for pydolfin.
#commonfile=os.path.join("site-packages", "pycc", "common.py")

if env["enablePydolfin"]:
    installfiles = scons.buildFileList(
        buildDataHash["pythonPackageDirs"])

    for f in installfiles:
        #installpath=os.path.sep.join(os.path.dirname(f).split(os.path.sep)[1:])
        env.Install(os.path.join(env["pythonModuleDir"],"dolfin"), f)

#env = scons.installCommonFile(env, commonfile, prefix)

# No data for dolfin?
_targetdir=os.path.join(prefix, "share", "dolfin", "data")
if 'install' in COMMAND_LINE_TARGETS and not os.path.isdir(_targetdir):
  os.makedirs(_targetdir)
env = scons.addInstallTargets(env, sourcefiles=buildDataHash["data"],
                              targetdir=_targetdir)

## No tests to install for dolfin?
## Not sure tests should be installed, have to check that.
#_targetdir=os.path.join(prefix, "share", "pycc", "tests")
#if 'install' in COMMAND_LINE_TARGETS and not os.path.isdir(_targetdir):
#  os.makedirs(_targetdir)
#env = scons.addInstallTargets(env, sourcefiles=buildDataHash["tests"],
#                              targetdir=_targetdir)

### I am not sure there are any docs to install with dolfin.
_targetdir=os.path.join(prefix, "share", "doc", "dolfin")
if 'install' in COMMAND_LINE_TARGETS and not os.path.isdir(_targetdir):
  os.makedirs(_targetdir)
env = scons.addInstallTargets(env, sourcefiles=buildDataHash["docs"],
                              targetdir=_targetdir)

# Instruct scons what to do when user requests 'install'
targets = [env["binDir"], env["libDir"], env["includeDir"], env["pkgConfDir"]]
if env["enablePydolfin"]:
    targets.append(env["pythonModuleDir"])
    targets.append(env["pythonExtDir"])
env.Alias("install", targets)

# _runTests used to use the global 'ret' (now buildDataHash). Therefore, we
# need to wrap _runTests in a closure, now that the functions is moved into
# 'scons'
_runTests = scons.gen_runTests(buildDataHash)

env.Command("runtests", buildDataHash["shlibs"] + buildDataHash["extModules"],
            Action(_runTests, scons._strRuntests))

# Create helper file for setting environment variables
f = open('dolfin.conf', 'w')
if env["PLATFORM"] == "darwin":
    f.write('export DYLD_LIBRARY_PATH="' + prefix + 'lib:$DYLD_LIBRARY_PATH"\n')
else:
    f.write('export LD_LIBRARY_PATH="'   + prefix + 'lib:$LD_LIBRARY_PATH"\n')
f.write('export PATH="'            + prefix + 'bin:$PATH"\n')
f.write('export PKG_CONFIG_PATH="' + prefix + 'lib/pkgconfig:$PKG_CONFIG_PATH"\n')
if env["enablePydolfin"]:
    pyversion=".".join([str(s) for s in sys.version_info[0:2]])
    f.write('export PYTHONPATH="'  + prefix + 'lib/python'    + pyversion + '/site-packages:$PYTHONPATH"\n')
f.close()

def help():
    # TODO: The message below should only be printed if there were no
    # errors running scons. In SCons 0.98 we can do this by checking
    # if GetBuildFailures() returns an empty list.
    #from SCons.Script import GetBuildFailures
    #if GetBuildFailures():
    #    # There have been errors. Write out error summary or just
    #    # return and let SCons handle the error message.
    #    return
    msg = """---------------------------------------------------------
If there were no errors, run

    scons install

to install DOLFIN on your system. Note that you may need
to be root in order to install. To specify an alternative
installation directory, run

    scons install prefix=<path>

You may also run ./scons.local for a local installation
in the DOLFIN source tree.

You can compile all the demo programs in the subdirectory
demo by running

   scons enableDemos=yes

---------------------------------------------------------"""
    print msg

def help_install():
    # TODO: The message below should only be printed if there were no
    # errors running scons. In SCons 0.98 we can do this by checking
    # if GetBuildFailures() returns an empty list.
    #from SCons.Script import GetBuildFailures
    #if GetBuildFailures():
    #    # There have been errors. Write out error summary or just
    #    # return and let SCons handle the error message.
    #    return
    #msg = """---------------------------------------------------------
#DOLFIN successfully compiled and installed in\n\n  %s\n""" % prefix
    # Check that the installation directory is set up correctly
    if not os.path.join(prefix,"bin") in os.environ["PATH"]:
        msg = """---------------------------------------------------------
Warning: Installation directory is not in PATH.

To compile a program against DOLFIN, you need to update
your environment variables to reflect the installation
directory you have chosen for DOLFIN. A simple way to do
this if you are using Bash-like shell is to source the
file dolfin.conf:

    source dolfin.conf

This will update the values for the environment variables
PATH, LD_LIBRARY_PATH, PKG_CONFIG_PATH and PYTHONPATH.
---------------------------------------------------------"""
        print msg
    #msg += "\n---------------------------------------------------------"
    #print msg

# Print some help text at the end
if not env.GetOption("clean"):
    import atexit
    if 'install' in COMMAND_LINE_TARGETS:
        atexit.register(help_install)
    else:
        atexit.register(help)

# Close log file
scons.logClose()

# vim:ft=python ts=2 sw=2
