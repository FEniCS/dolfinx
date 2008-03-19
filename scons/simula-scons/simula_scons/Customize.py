import SCons.Util, SCons.Scanner

import os, sys, re


def darwinCxx(env):
    """Change the cxx parts of env for the Darwin platform"""
    env["CXXFLAGS"] += " -undefined dynamic_lookup"
    env["SHLINKFLAGS"] += " -undefined dynamic_lookup"
    env["LDMODULEFLAGS"] += " -undefined dynamic_lookup"
    if not env.GetOption("clean"):
        # remove /usr/lib if present in LIBPATH.
        # there should maybe be some testing here to make sure 
        # it doesn't blow in our face.
        try:
            env["LIBPATH"].remove('/usr/lib')
        except:
            pass
    return env

def darwinSwig(env):
    """Change the swig parts of env for the Darwin platform"""
    env['SHLINKFLAGS'] = env['LDMODULEFLAGS']
    env['SHLINKCOM'] = env['LDMODULECOM']
    env["SHLIBSUFFIX"] = ".so"
    return env


def swigScanner(node, env, path):
    """What do I do? Arve will write this."""
    include_pattern = re.compile(r"%include\s+(\S+)")

    def recurse(path, search_path):
        """What do I do? Arve will write this."""
        f = open(path)
        try: contents = f.read()
        finally: f.close()

        found = []
        for m in include_pattern.finditer(contents):
            inc_fpath = m.group(1)
            # Strip off quotes
            inc_fpath = inc_fpath.strip("'").strip('"')
            real_fpath = os.path.realpath(inc_fpath)
            if os.path.dirname(real_fpath) != os.path.dirname(path):
                # Not in same directory as original swig file
                if os.path.isfile(real_fpath):
                    found.append(real_fpath)
                    continue

            # Look for unqualified filename on path
            for dpath in search_path:
                abs_path = os.path.join(dpath, inc_fpath)
                if os.path.isfile(abs_path):
                    found.append(abs_path)
                    break
        
        for f in [f for f in found if os.path.splitext(f)[1] == ".i"]:
            found += recurse(f, search_path)
        return found

    fpath = node.srcnode().path
    search_path = [os.path.abspath(d) for d in re.findall(r"-I(\S+)", " ".join(env["SWIGFLAGS"]))]
    search_path.insert(0, os.path.abspath(os.path.dirname(fpath)))
    r = recurse(fpath, search_path)
    return r

swigscanner = SCons.Scanner.Scanner(function=swigScanner, skeys=[".i"])

def swigEmitter(target, source, env):
    for src in source:
        src = str(src)
        if "-python" in SCons.Util.CLVar(env.subst("$SWIGFLAGS")):
            target.append(os.path.splitext(src)[0] + ".py")
    return (target, source)

class Dependency:

    def __init__(self, cppPath=None, libPath=None, libs=None, linkOpts=None, version=None, compiler=None):
        self.cppPath, self.libPath, self.libs, self.linkOpts, self.version, self.compiler  = cppPath, libPath, libs, linkOpts, version, compiler

    def __str__(self):
      return "\ncppPath: %s\nlibPath: %s\nlibs: %s\nlibs: %s\ncompiler: %s\n" % \
               (self.cppPath,self.libPath,self.libs,self.linkOpts,self.compiler)


