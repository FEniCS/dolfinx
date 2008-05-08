import re
import os.path
import datetime
import sys
from distutils import sysconfig

from SCons.Builder import Builder
from SCons.Action import Action
from SCons.Options import Options
from SCons.Environment import Environment

# Import module configuration logic
from _module import *
import pkgconfig
import Customize
from Errors import CommandError, PkgconfigGeneratorMissing

_defaultEnv=None

def setDefaultEnv(env):
    global _defaultEnv
    _defaultEnv=env

log_file = None

def logOpen(env):
    global log_file

    if env.has_key('SSLOG'):
        if env['SSLOG'] == '':
            log_file = sys.stdout
        elif log_file == None:
            log_file = open(env['SSLOG'], 'w+')
        elif log_file.closed:
            log_file = open(env['SSLOG'], 'a+')

def log(s):
    if log_file is not None:
        log_file.write("%s\n" % s)

def log2(s):
    if log_file is not None:
        calling_func_name = sys._getframe(1).f_code.co_name
        log_file.write("%s: %s\n" % (calling_func_name,s))

def logDate():
    log(datetime.datetime.now())

def logClose():
    if log_file != sys.stdout and log_file is not None:
        log_file.close()

def runCommand(command, args, captureOutput=True):
    """ Run command, returning its output on the stdout and stderr streams.

    If the command exits with a non-zero value, CommandError is raised.
    @param command: The name of the command
    @param args: The arguments to the command, either as an explicit sequence or as a whitespace delimited string.
    @return: A pair of strings, containing the command's output on stdout and stderr, respectively.
    """
    if isinstance(args, basestring):
        args = args.split()
    cl = "%s %s" % (command, " ".join(args))
    from subprocess import Popen, PIPE
    p = Popen(cl.split(), stdout=PIPE, stderr=PIPE, bufsize=-1)
    r = p.wait()
    out, err = p.communicate()
    if r:
        raise CommandError(cl, r, err.strip())
    if captureOutput:
        return out.strip(), err.strip()

def rsplit(toSplit, sub, max=None):
    s = toSplit[::-1] # Reverse string
    if max == None: l  = s.split(sub)
    else: l = s.split(sub, max)
    l.reverse() # Reverse list
    return [s[::-1] for s in l] # Reverse list entries

def oldrsplit(toSplit, sub, max):
    """ str.rsplit seems to have been introduced in 2.4 :( """
    l = []
    i = 0
    while i < max:
        idx = toSplit.rfind(sub)
        if idx != -1:
            toSplit, splitOff = toSplit[:idx], toSplit[idx + len(sub):]
            l.insert(0, splitOff)
            i = i + 1
        else:
            break

    l.insert(0, toSplit)
    return l

_SshHost = "gogmagog.simula.no"
if "SUDO_USER" in os.environ:
    # Use real user in case PyCC is install with 'sudo scons install'
    _SshHost = "%s@%s" % (os.environ["SUDO_USER"], _SshHost)
_SshDir = "/srl/phulius/pycc"

def sshList(path):
    """ Recursively list contents of directory on SSH server.
    @return: List of file paths relative to directory.
    """
    out, err = runCommand("ssh", [_SshHost, "ls", "-R", "%s/%s" % (_SshDir, path)])

    dirs = {}
    curDir = None
    for l in out.splitlines():
        l = l.strip()
        if not l:
            continue
        l = l.replace(_SshDir + "/data/", "")

        if l[-1] == ":":
            dname = l[:-1]
            if dname == _SshDir + "/data":
                continue
            if dname[0] == "/":
                dname = l[1:]
            curDir = dname
            dirs[curDir] = []
        else:
            if curDir is not None:
                dirs[curDir].append(l)

    paths = []
    for dname, entries in dirs.items():
        for e in entries:
            path = os.path.join(dname, e)
            if not path in dirs:
                paths.append(path)

    return paths

def _sshCopy(target, source, env):
    env = env.Copy()
    if os.environ.has_key("SSH_AGENT_PID"):
        env["ENV"]["SSH_AGENT_PID"] = os.environ["SSH_AGENT_PID"]
    if os.environ.has_key("SSH_AUTH_SOCK"):
        env["ENV"]["SSH_AUTH_SOCK"] = os.environ["SSH_AUTH_SOCK"]
    for tgt in target:
        tgt = str(tgt)
        env.Execute("scp -r %s:%s/%s %s" % (_SshHost, _SshDir, tgt, tgt))
def _sshDesc(target, source, env):
    tgts = [str(t) for t in target]
    return "Copying from SSH repository: %s" % ", ".join(tgts)
sshCopy = Builder(action=Action(_sshCopy, _sshDesc))

class Configure(object):
    """ Replace the standard SCons Configure method.
    
    The standard SCons Configure method is somewhat weak, in particular with
    regard to cleaning up after previous builds. When cleaning you normally
    don't want to reconfigure, you just want to clean the targets from the
    previous build. You may still want to get at configuration parameters
    however, since they may affect which targets are built (e.g. a module is
    only built in the presence of an external library). This class caches
    configuration values between builds, even command-line options, so that the
    build configuration when cleaning mirrors that when building.
    """
    def __init__(self, env, arguments, options=[], customTests={}):
        """ @param env: SCons environment
        @param arguments: SCons command-line arguments (ARGUMENTS)
        @param options: Command-line options.
        """
        useCache = env.GetOption("clean") # Prefer cached values if cleaning
        
        self._conf = env.Configure(log_file=os.path.join("scons", "configure.log"), custom_tests=\
                customTests)
        for n, f in customTests.items():
            setattr(self, n, getattr(self._conf, n))
        self._oldCache, self._newCache = {}, {}
        self._cachePath = os.path.abspath(os.path.join("scons", "configure.cache"))
        if useCache and os.path.exists(self._cachePath):
            try:
                f = open(self._cachePath)
                context = None
                for l in f:
                    m = re.match(r"(.+):", l)
                    if m:
                        context = m.group(1)
                        self._oldCache[context] = {}
                        continue
                    if context is None:
                        continue

                    m = re.match("(.+) = (.+)", l)
                    try:
                        k, v = m.groups()
                    except:
                        continue
                    self._oldCache[context][k] = v
            finally:
                f.close()

        args = arguments.copy()
        optsCache = os.path.join("scons", "options.cache")   # Save options between runs in this cache
        if useCache:
            # Then ignore new values
            for o in options:
                if o[0] in args:
                    del args[o[0]]

        opts = Options(optsCache, args=args)
        opts.AddOptions(*options)
        opts.Update(env)
        # Cache options if asked to
        cacheOptions = env.get("cacheOptions", False)
        if cacheOptions:
            del env["cacheOptions"] # Don't store this value
            # Wan't to make the 'veryClean' a on-off option, so we don't store
            veryCleanOption = env.get("veryClean", False)
            if veryCleanOption:
              del env["veryClean"] # Don't store the veryClean option
            opts.Save(optsCache, env)
            # Restore
            env["cacheOptions"] = cacheOptions
            if veryCleanOption:
              env["veryClean"] = veryCleanOption
        env.Help(opts.GenerateHelpText(env))
    
    def checkCxxHeader(self, header, include_quotes='""'):
        """ Look for C++ header.

        @param header: The header to look for. This may be a sequence, in which case the preceding
        items are headers to be included before the required header.
        """
        ctxt = "checkCxxHeader"
        try:
            r = self._oldCache[ctxt][str(header)]
        except KeyError:
            r = self._conf.CheckCXXHeader(header, include_quotes)

        if not ctxt in self._newCache:
            self._newCache[ctxt] = {}
        self._newCache[ctxt][str(header)] = r
        return r

    def checkLibWithHeader(self, lib, header, language="C", call=None):
        ctxt = "checkLibWithHeader"
        try:
            r = self._oldCache[ctxt][str(lib)]
        except KeyError:
            kwds = {}
            if call is not None:
                kwds["call"] = call
            r = self._conf.CheckLibWithHeader(lib, header, language, **kwds)

        if not ctxt in self._newCache:
            self._newCache[ctxt] = {}
        self._newCache[ctxt][str(lib)] = r
        return r

    def finish(self):
        """ Update configuration cache """
        f = open(self._cachePath, "w")
        try:
            for ctxt, d in self._newCache.items():
                f.write("%s:\n" % ctxt)
                for k, v in d.items():
                    f.write("%s = %s\n" % (k, v))
        finally:
            f.close()


## Gruff moved over from SConstruct
## should be cleaned up as well.
def setConstants(_dataLevelsBeneath, _DataDirRel):

    def setConstants(env, target, source):
        """Change the constants in target (from source) to reflect the location of
           the installed software."""
        
        tgt = str(target[0])
        src = str(source[0])
        f = open(src)
        txt = f.readlines()
        f.close()
        if not os.path.isdir(os.path.dirname(tgt)):
            os.makedirs(os.path.dirname(tgt))
        f = open(tgt, "w")
        try:
            found0 = found1 = False
            for l in txt:
                if l.startswith("_levelsBeneath = "):
                    l = "_levelsBeneath = %d\n" % _dataLevelsBeneath
                    found0 = True
                elif l.startswith("_dataDir ="):
                    l = '_dataDir = "%s"\n' % _DataDirRel
                    found1 = True
                f.write("%s" % l)
            if not found0 or not found1:
                raise Exception, "Failed to modify path to data directory in pycc.common"
        finally:
            f.close()

    return setConstants 


def pathValidator(key, val, env):
    """ Silence SCons with regard to non-existent paths """
    pass

def PathOption(dir, explanation, default):
    """ Define PathOption that works the same across old and new (0.96.9*) SCons, i.e. new SCons
    adds a restrictive path validator by default. 
    Used when setting commandline options in the Configure object.
    """
    return (dir, "%s ( /path/to/%s )" % (explanation, dir), default, pathValidator)

def defaultCxxCompiler():
    """Read the CXX environment variable if set"""
    if os.environ.has_key("CXX"):
        return os.environ["CXX"]
    else:
        return None

def defaultPythonLib(prefix=None, plat_specific=False):
  python_lib = sysconfig.get_python_lib(prefix=prefix, plat_specific=plat_specific)
  if not prefix is None:
    if not python_lib.startswith(prefix):
      python_lib = os.path.join(prefix,"lib","python" + sysconfig.get_python_version(),"site-packages")
  return python_lib

def defaultFortranCompiler():
    """Read the FORTRAN environment variable if set"""
    if os.environ.has_key("FORTRAN"):
        return os.environ["FORTRAN"]
    else:
        return None

def resolvePackages(candidates, defaultPackages):
    """Produce a complete list of packages.
    If a package in candidates in the basename (in front of the version-dash)
    match a basename among the defaultPackages, use the version from the
    candidates. Candidates that does not match anything in the defaultPackages 
    set are appended to the list of packages.
    """
    for p in defaultPackages:
        base = p[:p.find('-')]
        delIndex = -1
        for c in candidates:
            c_base = c[:c.find('-')]
            if c_base == base:
                defaultPackages[defaultPackages.index(p)] = c
                delIndex = candidates.index(c)
                break
        if delIndex != -1:
            del(candidates[delIndex])
    defaultPackages += candidates
    if defaultPackages.count(''):
        defaultPackages.remove('')
    return defaultPackages


def gen_runTests(dataHash):
    def _runTests(target, source, env):
        """Target 'runtests' installs everything and then executes runtests.py in
           the tests directory, with the environment set up as necessary"""

        ldPath, pyPath = [], [os.path.join(os.path.abspath("site-packages"), "pycc")]
        print pyPath
        for f in dataHash["shlibs"]:
            dpath = os.path.dirname(f.abspath)
            if dpath not in ldPath:
                ldPath.append(dpath)
        for f in dataHash["pythonModules"]:
            dpath = os.path.dirname(f.abspath)
            if dpath not in pyPath:
                pyPath.append(dpath)

        environ = os.environ.copy()
        environ["LD_LIBRARY_PATH"] = os.pathsep.join(ldPath + environ.get("LD_LIBRARY_PATH", "").split(os.pathsep))
        environ["PYTHONPATH"] = os.pathsep.join(pyPath + environ.get("PYTHONPATH", "").split(os.pathsep))

        oldDir = os.getcwd()
        os.chdir("tests")
        try:
            r = os.spawnlpe(os.P_WAIT, "python", "python", "runtests.py", environ)
        finally:
            os.chdir(oldDir)
        return r
    return _runTests

def _strRuntests(target, source, env):
    return "Running tests"

def qualifiedPathName(path):
    """Returns False if path contains a directory starting with dot. Else True"""

    for component in path.split(os.path.sep):
        if component[0] == ".": # omitt if a directory starts with .  
            return False
    return True

exclude_suffixes = (".pyc",".pyo",".tmp")

def qualifiedFileName(filename):
    """Returns False if filename starts with dot, or has illegal suffix.
    Illegal suffixes are defined in the exclude_suffixes module list.
    """
    if filename[0] == "." or os.path.splitext(filename)[1] in exclude_suffixes: 
        return False
    return True


def globFiles(dir, pattern,returnFile=True):
    from glob import glob
    if returnFile:
        return [_defaultEnv.File(f) for f in glob(os.path.join(dir, pattern))]
    else:
        return glob(os.path.join(dir, pattern))



#--------------------------------------------------------------------
# Figure out if and how to edit common.py
# Edit common.py
# Add installer for common.py to the environment
#--------------------------------------------------------------------
def installCommonFile(env, commonfile, prefix):
    if os.path.isfile(commonfile):

        RelativeDataDir = os.path.join("share", "pycc", "data")
        # Determine path to data from pycc.common module
        pyccPyPath = os.path.normpath(os.path.join(env.subst(env["pythonModuleDir"]), "pycc"))

        # Find out how to get from the pythonModuleDir to the data dir:
        commonprefix = os.path.commonprefix((prefix,pyccPyPath))
        prefixdiff = prefix[len(commonprefix):]
        RelativeDataDir = os.path.join(prefixdiff,RelativeDataDir)
        pyccPyPathRel = pyccPyPath[len(commonprefix):]
        _dataLevelsBeneath = len(pyccPyPathRel.split(os.path.sep))

        # create a callback for editing of common.py, in order to have
        # correct location of DataDir:
        callback = setConstants(_dataLevelsBeneath, RelativeDataDir)
        env.Alias("install", 
                env.Command(os.path.join("$pythonModuleDir", "pycc", "common.py"), 
                    commonfile, callback))
    return env

def buildFileList(dirs,glob="*.py",exclude=[]):
    """Build a list of python files to be installed in site-packages.
       Preserve the directory structure.
    """
    installfiles = []
    for p in dirs:
        for dpath, dnames, fnames in os.walk(p.path):

            srcPath = dpath.split(os.path.sep,1)[0]
            pyPath = os.path.sep.join(dpath.split(os.path.sep,1)[1:])

            if not qualifiedPathName(dpath): continue

            installfiles += globFiles(dpath,glob,returnFile=False)

    for excl in exclude:
        try: del(installfiles[installfiles.index(excl)])
        except: pass

    return installfiles

def addInstallTargets(env, sourcefiles=[], targetdir=""):
    """Add install targets to env from a list of files"""
    for f in sourcefiles:
        env.InstallAs(os.path.join(targetdir, f.path.split(os.path.sep, 1)[1]), f)
    env.Alias("install",targetdir)
    return env

def intlist(l):
    """Convert a list of strings to ints if possible. Leave the string 
    if not possible to convert.
    """
    return_l = []
    for item in l:
        try: item = int(item)
        except: pass
        return_l.append(item)
    return return_l


def checkVersion(v_check, v_required):
    """ Check that v_check is bigger that v_required """
    
    delimiters = re.compile(r"[._-]")

    vc = re.split(delimiters, v_check)
    vr = re.split(delimiters, v_required)

    vc = intlist(vc)
    vr = intlist(vr)

    length = min(len(vr), len(vc))
    for i in xrange(length):
        if vc[i] > vr[i]:
            return True
        elif vc[i] < vr[i]:
            return False
    # If the required version is the longest and we reach this place:
    # E.g.: checkVersion("1.2.15", "1.2.15.1") -> False
    if len(vr)-len(vc) > 0:
        return False
    # Identical versions, return True
    return True

# Find modules ( srcnode must be updated ) 
def getModules(configuredPackages, resolve_deps=True, directory="."):
    """Generate module objects from specification files in module directories. Directories starting with a dot,
       e.g. ".ShadowModule" will not be processed
       
       @param A dictionary of packages that is already configured
       @param A flag indicating whether dependencies should be resolved; if True, 
              modules with dependencies not present in configuredPackages will be 
              removed from the list
       @return A dictionary of L{module<_Module>} objects.
    """
    modules = {}
    srcDir = _defaultEnv.Dir(directory).srcnode().abspath
    if not srcDir[-1] == os.path.sep:
        srcDir += os.path.sep
    for dpath, dnames, fnames in os.walk(srcDir):
        dpath = dpath[len(srcDir):] # Make relative to source dir
        for d in dnames[:]:
            if d.startswith("."):
                # Ignore dirs starting with '.'
                dnames.remove(d)
                continue
            mpath = os.path.join(dpath, d)
            try: mod = readModuleConfig(os.path.join(srcDir, mpath), mpath, _defaultEnv)
            except NoModule:
                # No module configuration
                continue
            modules[mod.name] = mod

    # if no module found, try in given "directory":
    # Maybe we should do this always?
    # Finally, check for config file (scons.cfg) in the root of the project:
    try: 
      mod = readModuleConfig(srcDir, ".", _defaultEnv)
      modules[mod.name] = mod
    except NoModule:
      pass
        
    if resolve_deps:
        modules = resolveModuleDependencies(modules, configuredPackages)
    return modules

def getModulesAndDependencies(directory=".",
                              configuredPackages={}, disabled_packages=[]):
    """Find all modules and dependencies in the project.

    @return tuple with modules, external dependenices for each module, both 
            dicts, a hash with all dependencies listed, and a hash with the
            PkgConfig objects that are created.
    """

    # Find all modules:

    modules = getModules({}, resolve_deps=False, directory=directory)

    # next, build a list of all external dependencies for each module.
    # We also include the optional dependencies, as those are separated
    # inside the module anyway.

    dependencies = {}
    packcfgObjs = {}

    # all dependencies accumulated in one hash (will be returned as a list, but
    # it is simpler to build a uniq list in a hash, using .has_key)
    alldependencies = {}
    for module_name, module in modules.items():
        dependencies[module_name] = {}

        dict_deps = {}
        seq_deps = []

        # Dependencies - both mandatory and optional.
        if isinstance(module.dependencies, dict):
            dict_deps.update(module.dependencies)
        else:
            seq_deps += module.dependencies

        if isinstance(module.optDependencies, dict):
            dict_deps.update(module.optDependencies)
        else:
            seq_deps += module.optDependencies 

        if isinstance(module.swigDependencies, dict):
          dict_deps.update(module.swigDependencies)
        else:
          seq_deps += module.swigDependencies

        # Handle dependencies with a version requirement.
        for package,version in dict_deps.items():
            if not modules.has_key(package):

                # this is an external dependency
                if not dependencies[module_name].has_key(package):
                    dependencies[module_name][package] = version
                else:
                    # Dependency already registered. We update the information 
                    # either if the previous registration was with None as 
                    # version or an older version was requested earlier.
                    if not dependencies[module_name][package]:
                        dependencies[module_name][package] = version
                    elif checkVersion(version,dependencies[module_name][package]):
                        dependencies[module_name][package] = version

                # We also make sure that the package is present in 
                # alldependencies with the correct version info
                if not alldependencies.has_key(package):
                    # not present, add
                    alldependencies[package] = version
                else:
                    # already present, update version if necessary:
                    alldependencies[package] = dependencies[module_name][package]

        # Handle dependencies without any version requested.
        for package in seq_deps:
            if not modules.has_key(package):
                # this is an external dependency
                # Add in the dependencies with None as version if not previously 
                # listed
                if not dependencies[module_name].has_key(package):
                    dependencies[module_name][package] = None

                # Also, add in alldependencies if not already present:
                if not alldependencies.has_key(package):
                    alldependencies[package] = None

    # See which packages that can be resolved with PkgConfig right away
    # We can get a configuredPackages dict as input. If so, check that 
    # the package is not already there.
    for d in alldependencies.keys(): 
        if configuredPackages.has_key(d):
          continue
        if d in disabled_packages:
          # skip this dependency since it should be disabled
          continue
        print "Checking for %s..." % (d),
        try:
          packcfg = pkgconfig.PkgConfig(d, env=_defaultEnv)
          configuredPackages[d] = Customize.Dependency(cppPath=packcfg.includeDirs(), \
                                    libPath=packcfg.libDirs(),\
                                    libs=packcfg.libs(),\
                                    linkOpts=packcfg.linkOpts(),\
                                    version=packcfg.version(),\
                                    compiler=packcfg.compiler())
          packcfgObjs[d] = packcfg
        except:
          print "failed"
##           print "failed"
##           print """ *** Unable to generate a suitable pkg-config file for %s.
##  *** If %s is present on your system, try setting the %s_DIR 
##  *** environment variable to the directory where %s is installed.""" % (d,d,str.upper(d),d)
##           if os.environ.has_key("%s_DIR" % (str.upper(d))):
##             print " *** %s_DIR is currently set to %s" % (str.upper(d),os.environ["%s_DIR" % (str.upper(d))])

    return modules,dependencies,alldependencies,configuredPackages,packcfgObjs

def addToDependencies(deps,pack,ver=None):
    # Add new package into existing dependencies
    if isinstance(deps, dict):
        # original deps is a dict, with versions
        if deps.has_key(pack):
            # update if previously registered with None 
            if not deps[pack]:
                deps[pack] = ver
            # or if previously registered with older version
            elif checkVersion(ver,deps[pack]):
                deps[pack] = ver
        else:
            deps[pack] = ver
    else:
        # original deps is just a list, we append the new pack
        # if not already present. 
        # We don't care about version, that should be resolved before 
        # we add the new package.
        if pack not in deps:
            deps += [pack]

def resolveCompiler(configuredPackages, packcfgObjs, sconsEnv):
  """Find a common compiler for this package and it's external dependencies
      
     We have to make sure that this package and its dependencies can be compiled
     using a common compiler. An example is that PETSc may request mpicc if
     it was compiled with that one, while g++ is the default option for the
     current package. In that case, mpic++ or mpicxx should probably be used. 

     The pkg-config generators for external modules should provide a pkgTests 
     method that takes optionally a compiler, if the compiler is an important
     issue for the package. We call on this method to run tests with a 
     specific compiler

     pkg-config files for packages that needs a specific compiler should 
     include the compiler variable. The pkg-config opbject in the 
     configuredPackages hash will hold this compiler in the compiler 
     variable. The variable is None if none compiler is requested. 
  """
 
  # TODO: I should probably also handle linker here. Maybe we should assume 
  # compiler is a tuple with (compiler,linker) and if it is just a string, 
  # change it into (compiler,compiler) (i.e. assume that compiler and linker
  # is the same.
  # Or we can handle this in the pkgconfig generator, which is how it is done 
  # right now...
  print "Resolving compiler...",
  verifiedPackages = {}
  compiler = sconsEnv["CXX"]
  for pack, dep in configuredPackages.items():
    # If both compiler and dep.compiler is set:
    if dep.compiler and compiler:
      # if dep.compiler and compiler differ, we must do a full check.
      if dep.compiler != compiler:
        try:
          dep_packgen = pkgconfig.get_packgen(pack)
        except PkgconfigGeneratorMissing:
          log2("No pkg-config generator for package '%s'. Assuming Ok." % pack)
          verifiedPackages[pack] = dep
          continue
        # case 1: Try the new package with the old compiler
        # If ok, mark the new package as verified
        flag = True
        try:
          dep_packcfg = packcfgObjs[pack]
          dep_packgen.pkgTests(forceCompiler=compiler, sconsEnv=sconsEnv,
                               cflags=dep_packcfg.cflags(), libs=dep_packcfg.ldflags())
        except Exception, msg:
          log2("Some tests failed for package '%s'. Discarding package." % pack)
          log2(msg)
          flag = False
        if flag:
          verifiedPackages[pack] = dep
        # case 2: Try all the already verified packages with the new 
        # compiler. If everything is ok, use the new compiler, and mark
        # the new package as verified
        else:
          flag = True
          for vpack in verifiedPackages:
            try:
              vdep_packgen = pkgconfig.get_packgen(vpack)
            except PkgconfigGeneratorMissing:
              log2("No pkg-config generator for package '%s'. Assuming Ok." % vpack)
              continue
            try:
              vdep_packcfg = packcfgObjs[vpack]
              vdep_packgen.pkgTests(forceCompiler=dep.compiler, sconsEnv=sconsEnv,
                                    cflags=vdep_packcfg.cflags(), libs=vdep_packcfg.ldflags())
            except Exception, msg:
              log2("Some tests failed for package '%s'. Discarding package." % pack)
              log2(msg)
              flag = False
              break
          if flag:
            compiler = dep.compiler
            verifiedPackages[pack] = dep
      # if dep.compiler and compiler is the same, we just mark the package
      # as verified:
      else:
        verifiedPackages[pack] = dep

    # If compiler is set and dep.compiler is not, we have to check that the new
    # package works with the set compiler
    elif compiler and not dep.compiler:
      try:
        dep_packgen = pkgconfig.get_packgen(pack)
        try:
          dep_packcfg = packcfgObjs[pack]
          dep_packgen.pkgTests(forceCompiler=compiler, sconsEnv=sconsEnv,
                               cflags=dep_packcfg.cflags(), libs=dep_packcfg.ldflags())
          verifiedPackages[pack] = dep
        except Exception, msg:
          # Cannot compile, link, or run package. Discard it.
          log2("Some tests failed for package '%s'. Discarding package." % pack)
          log2(msg)
          continue
      except PkgconfigGeneratorMissing:
        # This dep does not have its own pkg-config generator. Treat as verified?
        log2("No pkg-config generator for package '%s'. Assuming Ok." % pack)
        verifiedPackages[pack] = dep

    # If dep.compiler is set and compiler is not, we have to check that the new 
    # compiler can be used with all the verified packages. If so, use the new 
    # one as default and mark the package is verified, if not ditch the new 
    # package (and the compiler)
    elif dep.compiler and not compiler:
      flag = True
      for vpack in verifiedPackages:
        try:
          vdep_packgen = pkgconfig.get_packgen(vpack)
        except PkgconfigGeneratorMissing:
          # consider this dependency as verified
          log2("No pkg-config generator for package '%s'. Assuming Ok." % pack)
          continue
        try:
          vdep_packcfg = packcfgObjs[vpack]
          vdep_packgen.pkgTests(forceCompiler=dep.compiler, sconsEnv=sconsEnv,
                                cflags=vdep_packcfg.cflags(), libs=vdep_packcfg.ldflags())
        except Exception, msg:
          log2("Some tests failed for package '%s'. Discarding package." % pack)
          log2(msg)
          flag = False
          break
      if flag:
        compiler = dep.compiler
        verifiedPackages[pack] = dep
  print "done"
  # Store the compiler in the SCons environment and return the verifiedPackages
  # hash as the new configuredPackages
  if sconsEnv["CXX"] != compiler:
    print " Some tests failed using %s" % sconsEnv["CXX"]
    print " Switching to use %s instead." % compiler
  sconsEnv['CXX'] = compiler
  return verifiedPackages

def circularCheck(modlist, checkModName, allmodules):
  # mod.dependencies is either a dict (if versions are specified), else
  # it is just a list.
  #
  # For our purpose, though, we only care about the names of 
  # the dependencies. 
  #
  # When we traverse thorugh, we may stumble upon circular dependencies further 
  # down in the tree - not only for the checkModName, but actually for
  # any node (module). Hence, we need to aggregate the nodes as we go down in the 
  # tree. So, checkModName should be a list.
  #
  # We're doing a depth-first search.

  # First see if any of the modules in modlist match any of entries in 
  # the checkModName:
  if len([m for m in checkModName if m in modlist]) > 0:
    return True
  else:
    # only care about the modules in the allmodules (internal) list.
    nextlevel_modlist = [m for m in modlist if m in allmodules]
    # dig out the dependencies from each of the modules and the 
    # nextlevel_modlist, and check those:
    for modName in nextlevel_modlist:
      mod = allmodules[modName]
      # need to append modName to the checkModName, but only for this call - 
      # if we get back from this subtree in False state (no circularity
      # detected), we should not keep modName on the list. 
      if circularCheck(mod.dependencies, checkModName + [modName], allmodules):
        return True
  return False

def resolveModuleDependencies(modules, configuredPackages, packcfgObjs, 
                              internalmodules=None, sconsEnv=_defaultEnv):
    """Check whether dependencies can be fullfilled
    
       For each module, check whether external dependency requirements, 
       given in the 'deps' part, can be fullfilled in the packages configured.
       If some dependency is given with an explicit version requirement, check
       that the configured package agree with this requirement.

       Arguments:

         modules; a dict with detected modules, which should be resolved
         configuredPackages; a dict of known external packages
         internalmodules; if not None, a dict with known internal modules
           that 'modules' should be resolved against. 
           Use case: when building tests, we need to resolve against the
           library source modules, not the test modules. Hence we can pass
           in internalmodules as detected in the library sources.
           By default, modules is used as internalmodules.
         sconsEnv; SCons environment that may be modified with a new 'CXX'
           compiler.

       A new modules dict and the modified SCons environment are returned.
    """

    # Find a common compiler for this package and it's external dependencies
    if not sconsEnv.has_key("enableResolveCompiler") or \
           sconsEnv["enableResolveCompiler"]:
        configuredPackages = resolveCompiler(configuredPackages, packcfgObjs, sconsEnv)
    
    if not internalmodules:
      internalmodules = modules

    # First, we like to take care of circular internal dependencies, which we 
    # do not allow, of any order:
    # If module A, have deps d = [d1,d2,d3,..], A could not be among the deps
    # for any of the modules in d. Further on, none of the deps in d must 
    # themself have deps wich in turn include the module A. And so on...

    deleteModules = []
    for modName, mod in internalmodules.items():
      if circularCheck(mod.dependencies, [modName], internalmodules):
        # have to remove 'modName' from the modules that should be 
        # compiled, as a True return from circularCheck indicate that 
        # modName is found among the mod.dependencies, or their dependencies,
        # recursively. But we are not allowed to do that during interation:
        #
        # TODO: check logic here, is it flawed. If we have circular dependencies,
        # all modules that are involved in the circle must be removed? Or
        # will that be taken care of below?
        deleteModules.append(modName)

    for m in deleteModules:
      del internalmodules[m]
      # also remove this modName from the modules, as those are what we
      # try to resolve:
      if modules.has_key(m):
        del modules[m]

    new_modules = {}
    check_modules = modules.keys()
    
    # should ensure that check_modules is not an empty list, else the pop 
    # will raise an exception. But, if check_modules is empty, we probably
    # have more important problems already? (at this stage, anyway)
    modName = check_modules.pop(0) 
    mod = internalmodules[modName] # grab the actual module.

    check_more = True
    while check_more: 
      module_ok = True
      # TODO: Do we have to consider whether dependencies are list or dict 
      # here?
      for d in mod.dependencies:
        if d in internalmodules:
          if d in check_modules:
            # d not checked yet, postpone check of mod
            check_modules.append(modName)
            module_ok = False # Can not say that module is ok yet
                              # but that will be tested later on
            if len(check_modules) > 0:
              modName = check_modules.pop(0)
              mod = internalmodules[modName]
            else:
              check_more = False
            break # exit for-loop, no need to test other deps.
          elif new_modules.has_key(d):
            # d is checked and ok, we can just move to next in for-loop
            continue
          else:
            # d is checked and not verified -> throw away mod:
            module_ok = False
            print "Warning: Dependency %s will not be compiled" % (d)
            print "     ->  Ignoring module %s" % (modName)
            if len(check_modules) > 0:
              modName = check_modules.pop(0)
              mod = internalmodules[modName]
            else:
              check_more = False
            break # exit for-loop, no need to test other deps.
        elif configuredPackages.has_key(d):
          # d is an external module
          # Need to handle both the situation with deps. a dict (with
          # versions) and deps as a list... (see below)
          if isinstance(mod.dependencies, dict):
            # dependencies given with version, have to check against 
            # version in configuredPackages, unless the version requested for this
            # particular dependency was just given as None - meaning we don't care.
            request_version = mod.dependencies[d]
            if request_version:
              if not checkVersion(configuredPackages[d].version, request_version):
                # We can not compile module - 
                module_ok = False
                print "Warning: Requested version %s for package %s is not fullfilled" % (request_version,d)
                print "     ->  Ignoring module %s" % (modName)
                if len(check_modules) > 0:
                  modName = check_modules.pop(0)
                  mod = internalmodules[modName]
                else:
                  check_more = False
                break # exit for-loop, no need to test other deps.
        else:
          # The dependency that was asked for is neither among the internal modules
          # nor the configured external packages. Hence we don't know anything about 
          # it and have to throw this module away.
          module_ok = False
          print "Warning: Unknown dependency package: %s" % (d)
          if len(check_modules) > 0:
            modName = check_modules.pop(0)
            mod = internalmodules[modName]
          else:
            check_more = False
          break # exit for-loop, no need to test other deps.

      # we're done with the for-loop. If we get here with module_ok True, 
      # the module can be compiled and should go on into verified_modules
      # dict.
      if module_ok:
        new_modules[modName] = mod
        if len (check_modules) > 0:
          modName = check_modules.pop(0)
          mod = internalmodules[modName]
        else:
          check_more = False

#    for modName, mod in modules.items():
#
#        #print "*** Check deps for %s ***" % (modName)
#        if isinstance(mod.dependencies, dict):
#           
#            # for each dependency, we need to check whether the dep. is between 
#            # the configuredPackages.
#            # If it is, we need to check the version
#
#            compileModule = True
#            for package, request_version in mod.dependencies.items():
#                if configuredPackages.has_key(package):
#                    #print "package %s is configured" % (package)
#                    #print "request version: ",request_version
#                    #print "found current configured and using version: ",configuredPackages[package].version
#                    if request_version: # requested version is not None
#                        if not checkVersion(configuredPackages[package].version, request_version):
#                            compileModule = False
#                            print "Warning: Requested version %s for package %s is not fullfilled" % (request_version,package)
#                elif not internalmodules.has_key(package):
#                    compileModule = False
#                    print "Warning: Unknown dependency package: %s" % (package)
#                else:
#                    pass # Internal dependency; no need to check version.
#
#            if not compileModule:
#                print "     ->  Ignoring module %s" % (modName)
#            else:
#                mod.dependencies = mod.dependencies.keys()
#                new_modules[modName] = mod
#        else:
#            compileModule = True
#            for package in mod.dependencies:
#                if not configuredPackages.has_key(package) \
#                   and not internalmodules.has_key(package):
#                   compileModule = False
#                   print "Warning: Unknown dependency package: %s" % (package)
#
#            if not compileModule:
#                print "     ->  Ignoring module %s" % (modName)
#            else:
#                new_modules[modName] = mod

    # for the remaining modules we need to check whether optional dependencies
    # was specified, and if so se if we can resolve these. If we can resolve
    # them, we have to add '-DHAS_<module>' to cxx-flags.
    found_packages = []
    not_found_packages = []
    for modName,mod in new_modules.items():
        if isinstance(mod.optDependencies, dict):
            for package,request_version in mod.optDependencies.items():
                found = False
                if configuredPackages.has_key(package):
                    if request_version: # requested version is not None
                        if checkVersion(configuredPackages[package].version, request_version):
                           # Add cxxflags, and add package as a regular dependency.
                            mod.cxxFlags += " -DHAS_%s=1" % (package.upper())
                            mod.swigFlags.append("-DHAS_%s=1" % (package.upper()))
                            addToDependencies(mod.dependencies,package)
                            found = True
                    else:
                        # Add cxxflags, and add package as a regular dependency.
                        mod.cxxFlags += " -DHAS_%s=1" % (package.upper())
                        mod.swigFlags.append("-DHAS_%s=1" % (package.upper()))
                        addToDependencies(mod.dependencies,package)
                        found = True
                elif internalmodules.has_key(package):
                    # Add cxxflags, and add package as a regular dependency.
                    mod.cxxFlags += " -DHAS_%s=1" % (package.upper())
                    mod.swigFlags.append("-DHAS_%s=1" % (package.upper()))
                    addToDependencies(mod.dependencies,package)
                    found = True
                else:
                    found = False
                if not found:
                    not_found_packages.append("%s (version %s)" % (package,request_version))
                else:
                    found.append("%s (version %s)" % (package,request_version))
        else:
            for package in mod.optDependencies:
                found = False
                if configuredPackages.has_key(package):
                    # Add cxxflags, and add package as a regular dependency.
                    mod.cxxFlags += " -DHAS_%s=1" % (package.upper())
                    mod.swigFlags.append("-DHAS_%s=1" % (package.upper()))
                    addToDependencies(mod.dependencies,package)
                    found = True
                elif internalmodules.has_key(package):
                    # Add cxxflags, and add package as a regular dependency.
                    mod.cxxFlags += " -DHAS_%s=1" % (package.upper())
                    mod.swigFlags.append("-DHAS_%s=1" % (package.upper()))
                    addToDependencies(mod.dependencies,package)
                    found = True
                else:
                    found = False
                if not found:
                    not_found_packages.append(package)
                else:
                    found_packages.append(package)

    for package in found_packages:
        print "Found optional package: %s" % package
    for package in not_found_packages:
        print "Unable to find optional package: %s" % package

    return new_modules, sconsEnv

# vim:ft=python sw=2 ts=2
