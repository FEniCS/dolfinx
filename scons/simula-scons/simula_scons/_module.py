import os.path

class NoModule(Exception):
    """ No module defined for this directory """
    pass

class InvalidDefinition(Exception):
    """ A parameter is invalidly defined. """

class InvalidLibSources(InvalidDefinition):
    pass

class InvalidLibHeaders(InvalidDefinition):
    pass

class InvalidSwigSources(InvalidDefinition):
    pass

class InvalidProgSources(InvalidDefinition):
    pass

class InvalidDependencies(InvalidDefinition):
    pass

class InvalidSwigDependencies(InvalidDefinition):
    pass

class InvalidCxxFlags(InvalidDefinition):
    def __init__(self, value):
        InvalidDefinition.__init__(self, "Invalidly specified C++ compiler flags: %r" % (value,))

class InvalidSwigFlags(InvalidDefinition):
    def __init__(self, value):
        InvalidDefinition.__init__(self, "Invalidly specified swig flags: %r" % (value,))

class InvalidLinkFlags(InvalidDefinition):
    pass

class InvalidPowerStr(InvalidDefinition):
    pass

def _checkSequence(parameter, excType):
    if parameter is None:
        return []
    if not isinstance(parameter, (list, tuple)):
        raise excType(parameter)
    return parameter

def _checkDictionary(parameter, excType):
    if parameter is None:
        return {}
    if not isinstance(parameter, dict):
        raise excType(parameter)
    return parameter

class _Module:
    """ Module representation.

    @ivar name: The module's name.
    @ivar path: Path to the module directory.
    @ivar libSources: Any sources for building a library.
    @ivar swigSources: Any sources for building a swig wrapper.
    @ivar progSources: A dictionary listing sources for building any executable programs private to \
    the module (i.e. for testing purposes).
    @ivar dependencies: The modules dependencies, names of either other modules or external libraries.
    @ivar swigDependencies: Dependencies for the module's SWIG wrapper.
    @ivar cxxFlags: The module's C++ flags.
    @ivar linkFlags: The module's linker flags.
    @ivar swigFlags: Extra flags for swig
    @ivar powerStr: A custom string that is executed when building the module environment.
          You can access the module environment directely with 'modEnv'

    A module will also know how to create a pkg-config files for itself.
    """
    def __init__(self, name, fullpath, path, libSources, libHeaders, swigSources, progSources, dependencies, \
            optDependencies, swigDependencies, cxxFlags, linkFlags, swigFlags, powerStr):
        for attr in ("name", "fullpath", "path", "libSources", "libHeaders", "swigSources", "progSources", \
                     "dependencies", "optDependencies", "swigDependencies", "cxxFlags", "linkFlags", "swigFlags",\
                     "powerStr"):
            setattr(self, attr, locals()[attr])

    def pkg_config_generator(self, env, modules, configuredPackages):
      """A pkg-config file generator for this module. Projectname is taken from env      
      The modules argument is a pointer to the full modules dictionary, and 
      configuredPackages is external known packages. We need this to get
      the dependencies right.
      """

      # create a pkg-config file named env["projectname"]_self.name.pc, located
      # in #/projectname/self.path/
      # TODO: check that the description is accurate.
      # Return the created file using SCons abs.path syntax ('#' as root)
      modpcfile = os.path.join(self.path,"%s_%s.pc.in" % (env["projectname"],self.name))
      mod_reqs = []
      for d in self.dependencies:
        if d in modules:
          # internal dependency, prepend the projectname
          mod_reqs.append("%s_%s" % (env["projectname"],d))
        else:
          # external dependency
          mod_reqs.append(d)
      for d in self.optDependencies:
        if d in configuredPackages:
          # optional dependency exist, add:
          mod_reqs.append(d)
      mod_reqs = " ".join(mod_reqs)
      mod_libs = "-l%s" % (self.name)

      # compiler:
      # if env["CXX"] is set (are we sure we always use C++ - probably not...)
      if env.has_key("CXX") and env["CXX"] != None:
        compiler=env["CXX"]
      else:
        compiler=""

      modpc_str = """prefix=%s
exec_prefix=${prefix}
includedir=${prefix}/include
compiler=%s
libdir=${exec_prefix}/lib

Name: %s_%s
Version: 0
Description: The %s module in the %s project
Requires: %s
Cflags: -I${includedir}
Libs: -L${libdir} %s
""" % (env["prefix"], compiler, env["projectname"], self.name, self.name, env["projectname"], mod_reqs, mod_libs)
      f = open(modpcfile,'w')
      f.write(modpc_str)
      f.close()

      return os.path.join("#%s" % env["projectname"], modpcfile)

def readModuleConfig(dirPath, modulePath, env):
    """ Read module configuration script.

    A module configuration is read from a file 'scons.cfg' (or 'scons.local.cfg' if it exists),
    containing parameters pertaining to a module's build configuration. It is possible to
    specify both full dependencies (on headers and libraries) and just headers (need the include path).
    @param dirPath: Path to directory, presumably containing module configuration.
    @param modulePath: Path to module directory relative to source root.
    @return: A L{module<_Module>} object
    """
    import os.path, glob
    modName = os.path.basename(dirPath)
    modName = modName.replace("-", "_")
    cxxFlags = [f.strip() for f in env["CXXFLAGS"].split()]
    linkFlags = [f.strip() for f in str(env["LINKFLAGS"]).split()]
    if env.has_key("SWIGFLAGS") and env["SWIGFLAGS"]:
        swigFlags = [f.strip() for f in env["SWIGFLAGS"].split()]
    else:
        swigFlags = []
    ns = {"ModuleName": modName, "CxxFlags": cxxFlags, "LinkFlags": linkFlags, "SwigFlags": swigFlags}
    exec "import os.path; from glob import glob; from distutils import sysconfig" in ns
    try:
        exec "FORTRAN='%s'" % (env["FORTRAN"]) in ns 
    except:
        exec "FORTRAN='None'" in ns

    # Allow 'Default' to be used as an alias for 'None' in scons.cfg files
    # 'Default' will be used to represent no particular version in dependencies.
    ns["Default"] = None

    # Look for scons.local.cfg first:
    cfgPath = os.path.join(dirPath, "scons.local.cfg")
    if not os.path.isfile(cfgPath):
        # No scons.local.cfg file. Look for scons.cfg instead:
        cfgPath = os.path.join(dirPath, "scons.cfg")
        if not os.path.isfile(cfgPath):
            raise NoModule

    f = file(cfgPath)
    try:
        oldDir = os.getcwd()
        try:
            os.chdir(dirPath)
            exec f in ns
        finally:
            os.chdir(oldDir)
    finally: f.close()

    modName = ns.get("ModuleName", modName)
    libSources = _checkSequence(ns.get("LibSources", []), InvalidLibSources)
    libHeaders = ns.get("LibHeaders", None)
    if libHeaders is None:
        # Default to source/header pairs
        libHeaders = ["%s.h" % os.path.splitext(f)[0] for f in libSources]
    else:
        libHeaders = _checkSequence(libHeaders, InvalidLibHeaders)

    swigSources = ns.get("SwigSources", None)
    if swigSources is None:
        # Default to .i files found in "swig" subdirectory
        swigDir = os.path.join(dirPath, "swig")
        if os.path.isdir(os.path.join(dirPath, "swig")):
            swigSources = [os.path.join("swig", "%s.i" % modName)]
        else:
            swigSources = []
    else:
        swigSources = _checkSequence(swigSources, InvalidSwigSources)

    #progSources = _checkDictionary(ns.get("ProgSources", {}), InvalidProgSources)
    progSources = _checkDictionary(ns.get("AppSources", {}), InvalidProgSources)

    # Part of a transition - Dependencies will in the future be allowed to be
    # a dictionary, to state both a package and a required version
    try: 
        deps = _checkDictionary(ns.get("Dependencies", None), InvalidDependencies)
    except:
        deps = _checkSequence(ns.get("Dependencies", None), InvalidDependencies)

    swigDeps = _checkSequence(ns.get("SwigDependencies", []), InvalidSwigDependencies)

    # We may have optional dependencies. 
    # If an optional dependency is found on the compiling system, an extra 
    # compileflag -DHAS_<dependency> should be set. It is up to the implementer
    # of the module to make sure that things behave correctly - that is, system
    # should work both with and without optional packages. The format of 
    # OptDependencies is the same a the other deps.

    try:
      optDeps = _checkDictionary(ns.get("OptDependencies", []), InvalidDependencies)
    except:
      optDeps = _checkSequence(ns.get("OptDependencies", None), InvalidDependencies)

    cxxFlags = _checkSequence(ns.get("CxxFlags", cxxFlags), InvalidCxxFlags)
    cxxFlags = " ".join(cxxFlags)
    linkFlags = _checkSequence(ns.get("LinkFlags", linkFlags), InvalidLinkFlags)

    swigFlags = _checkSequence(ns.get("SwigFlags", swigFlags), InvalidSwigFlags)

    powerStr  = _checkSequence(ns.get("PowerStr", None), InvalidPowerStr)
    
    mod = _Module(modName, dirPath, modulePath, libSources, libHeaders, swigSources, progSources, deps, optDeps, swigDeps, cxxFlags, linkFlags, swigFlags, powerStr)
    return mod
