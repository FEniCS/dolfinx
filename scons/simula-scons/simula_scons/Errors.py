""" A few useful exceptions """
import simula_scons as scons

def _configError(msg):
    sys.stderr.write("%s\n" % msg)
    Exit(1)

class TestError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, "Test failed: %s" % msg)

class CommandError(Exception):
    def __init__(self, command, exitVal, stderr):
        Exception.__init__(self, "Command `%s' exited with value: %d" % (command, exitVal))
        self.stderr = stderr

class PkgconfigError(Exception):
    def __init__(self, pkg, msg):
        Exception.__init__(self, msg)
        self.lib = scons.rsplit(pkg, "-", 1)[0]

class PkgconfigMissing(PkgconfigError):
    def __init__(self, pkg):
        PkgconfigError.__init__(self, pkg, "The library '%s' couldn't be found, consider creating a pkg-config entry for it (`%s')" %
                (pkg.split("-", 1)[0], pkg + ".pc"))

class PkgconfigGeneratorMissing(PkgconfigError):
    def __init__(self, pkg):
        PkgconfigError.__init__(self, pkg, "A pkg-config generator for '%s' couldn't be found, consider creating a pkg-config entry ('%s') manually" %
                (pkg.split("-", 1)[0], pkg + ".pc"))

class PkgconfigGeneratorsMissing(Exception):
    def __init__(self):
        Exception.__init__(self, "The pkg-config generators can not be found. Please check your installation")

class ConfigMissing(Exception):
    def __init__(self, pkg, msg=""):
        Exception.__init__(self,"The config for library '%s' could not be found. %s" % (pkg,msg))



