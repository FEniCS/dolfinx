#
# Various functionality for handling pkg-config in a scons based build.
#
# The "public interface" here is the 'generate' method and the 'PkgConfig' 
# class. The 'generate' method can be used to generate a pkg-config file 
# based on a template. Within pycc this is used to generate a suiteable
# pkg-config file for the version of pycc currently build. 
# The 'PkgConfig' class is used to read pkg-config files using the pkg-config
# command, and output the result suitable for inclusion in scons build 
# environments.

import os, os.path, re, sys

# imports from the global 'SCons'
from SCons import Builder, Action

# import the local 'scons'
import simula_scons as scons
from simula_scons.Errors import CommandError, PkgconfigError, PkgconfigMissing, PkgconfigGeneratorsMissing, PkgconfigGeneratorMissing

def generate_pcFunc(replaceDict):
  def _pcFunc(target, source, env):
    """ Fill in .pc template. """
    f = file(str(source[0]))
    try: lines = f.readlines()
    finally: f.close()

    includeDir = env["includeDir"]
    libDir = env["libDir"]
    prefix = env["prefix"]
    f = file(str(target[0]), "w")

    # compiler:
    # if env["CXX"] is set (are we sure we always use C++ - probably not...)
    if env.has_key("CXX") and env["CXX"] != None:
      compiler=env["CXX"]
    else:
      compiler=""

    try:
      includeDir = includeDir.replace("$prefix", "${prefix}")
      libDir = libDir.replace("$prefix", "${exec_prefix}")
      header = ["prefix=%s\n" % prefix, "exec_prefix=${prefix}\n", "includedir=%s\n" % includeDir, \
              "libdir=%s\n" % libDir, "compiler=%s\n" % compiler]
      for i in range(len(lines)):
          if not lines[i].strip():
              break
      body = lines[i:]

      # Might need to susbstitute things in the body, e.g. dependencies.

      if replaceDict is not None:
        for item in replaceDict:
          # find line(s) in body that match the substitution key:
          replLines = [ l for l in body if "@"+item+"@" in l ]
          # replace the key with the replaceDict item, and feed into the body
          # We support multiple occurences of a string, although that should be 
          # rare:
          for l in replLines:
            body[body.index(l)] = l.replace("@"+item+"@", replaceDict[item])

      f.writelines(header + body)
    finally:
      f.close()
  return _pcFunc

def _strFunc(target, source, env):
    return "Building %s from %s" % (target[0], ", ".join([str(s) for s in source]))
  

def generate(env, replace=None):
    """Place a template-based generator for pkg-config files in the BUILDERS."""
    # we might need to pass in things to replace in the pkg-config, we
    # can use the replace as a dict for that. The keys will be things to
    # replace in the template and the values will be the replacements.
    # 
    # We need to access the replace-dict inside _pcFunc, hence we must turn the 
    # _pcFunc into a closure.
    _pcFunc = generate_pcFunc(replace)
    env["BUILDERS"]["PkgConfigGenerator"] = Builder.Builder(action={".in": Action.Action(_pcFunc, \
            strfunction=_strFunc)}, suffix="")


def get_packgen(package):
  # Try to import and run the right pkgconfig generator:
  try:
    packgen = __import__("simula_scons.pkgconfiggenerators",globals(),locals())
  except:
    raise PkgconfigGeneratorsMissing()
  # Generate the pkgconfig file:
  # There should probably be a second try/except around the import of the 
  # pkgconfig generator for the specific module.
  ns = {}
  try:
    exec "from simula_scons.pkgconfiggenerators import %s" % (package.split('-',1)[0]) in ns
  except:
    raise PkgconfigGeneratorMissing(package)
  packgen = ns.get("%s" % (package.split('-',1)[0]))
  return packgen


class PkgConfig(object):
    """Handling of pkg-config files.

       Whenever pkg-config file must be read to handle a dependency, an 
       instance of this class will be created.
    """
    def __init__(self, package, env):
      
      # If the PKG_CONFIG_PATH variable is empty, we are probably on 
      # deep water here.
      # I'll create a suiteable directory and set as PKG_CONFIG_PATH 
      # right away.

      # Set up a temporary place for pkgconfig files. 
      pkgconfdir = os.path.join(env.Dir("#scons").abspath,"pkgconfig")
      # set this is SCONS_PKG_CONFIG_DIR, which should be the place the 
      # pkgconfiggenerators put pc-files during build
      os.environ["SCONS_PKG_CONFIG_DIR"] = pkgconfdir
      # Make sure that the directory exist:
      if not os.path.isdir(pkgconfdir):
        os.makedirs(pkgconfdir)
      
      if not os.environ.has_key("PKG_CONFIG_PATH") or os.environ["PKG_CONFIG_PATH"] == "":
        os.environ["PKG_CONFIG_PATH"]=pkgconfdir
        #print "\n** Warning: Added %s \n    as PKG_CONFIG_PATH **" % (pkgconfdir)
      elif os.environ.has_key("PKG_CONFIG_PATH") and pkgconfdir not in os.environ["PKG_CONFIG_PATH"]:
        pkgConfPath = os.environ["PKG_CONFIG_PATH"].split(os.path.pathsep)
        pkgConfPath.append(pkgconfdir)
        os.environ["PKG_CONFIG_PATH"] = os.path.pathsep.join(pkgConfPath)
        #print "\n** Warning: Added %s \n    in PKG_CONFIG_PATH **" % (pkgconfdir)
      
      self.package = package
      self.env = env
      try: 
        scons.runCommand("pkg-config", ["--exists", self.package])
        print "yes"
      except CommandError:
        print "no (pkg-config file not found)"
        print " Trying to generate pkg-config file for %s..." % (self.package),

        # Construct pkgconfig-file
        packgen = get_packgen(package)

#        # Try to import and run the right pkgconfig generator:
#        try:
#          packgen = __import__("simula_scons.pkgconfiggenerators",globals(),locals())
#        except:
#          raise PkgconfigGeneratorsMissing()
#        # Generate the pkgconfig file:
#        # There should probably be a second try/except around the import of the 
#        # pkgconfig generator for the specific module.
#        ns = {}
#        exec "from simula_scons.pkgconfiggenerators import %s" % (package.split('-',1)[0]) in ns
#        packgen = ns.get("%s" % (package.split('-',1)[0]))

        packgen.generatePkgConf(sconsEnv=env)


    def _pkgconfig(self, param):
        #os.environ["PKG_CONFIG_ALLOW_SYSTEM_CFLAGS"] = "1"
        #os.environ["PKG_CONFIG_ALLOW_SYSTEM_LIBS"] = "1"
        try: out, err = scons.runCommand("pkg-config", [param, self.package])
        except CommandError, err:
            raise PkgconfigError(self.package, "Error reported by pkg-config: `%s'" % err.stderr)
        return out

    def version(self):
        """Find out what version the package think it is"""
        return self._pkgconfig("--modversion")

    def includeDirs(self):
        out = self._pkgconfig("--cflags-only-I")
        dirs = []
        opts = out.split()
        for o in opts:
            dirs.append(o[2:])
        return dirs

    def libDirs(self):
        out = self._pkgconfig("--libs-only-L")
        dirs = []
        opts = out.split()
        for o in opts:
            dirs.append(o[2:])
        return dirs

    def linkOpts(self):
      link_opts = self._pkgconfig("--libs-only-other").split()
      return link_opts

    def compiler(self):
      # If the compiler, and maybe the compilertype variable is set, read and 
      # return as a tuple. If I can't figure out the compilertype, I use None
      # to denote 'unknown'
      compiler = self._pkgconfig("--variable=compiler")
      if compiler == "":
        return None
      else:
        return compiler

    def frameworks(self):
        out = self._pkgconfig("--libs")
        fw = re.findall(r"-framework (\S+)",out)
        return fw

    def libs(self):
        """return a set of libraries for lib. 
           On Darwin (MacOSX) a tuple with libraries and frameworks is returned
        """
        out = self._pkgconfig("--libs-only-l")
        libs = []
        opts = out.split()
        for o in opts:
            libs.append(o[2:])
        if self.env["PLATFORM"] == "darwin":
            return (libs, self.frameworks())
        return libs, []

    def cflags(self):
      return self._pkgconfig("--cflags")

    def ldflags(self):
      return self._pkgconfig("--libs")
