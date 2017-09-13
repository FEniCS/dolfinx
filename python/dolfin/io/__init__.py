import dolfin.function
import dolfin.cpp as cpp


# Functions to extend cpp.io.File with

def __lshift__(self, u):
    """Support for the legacy '<<' output to file syntax"""

    # Note: __lshift__ notation for IO will be deprecated, see
    # https://bitbucket.org/fenics-project/dolfin/issues/895.

    if isinstance(u, dolfin.function.function.Function):
        self.write(u._cpp_object)
    elif isinstance(u, tuple):
        if isinstance(u[0], dolfin.function.function.Function):
            self.write(u[0]._cpp_object, u[1])
        else:
            self.write(u[0], u[1])
    else:
        self.write(u)


def __rshift__(self, u):
    """Support for the legacy '>>' read from file syntax"""

    # Note: __rshift__ notation for IO will be deprecated, see
    # https://bitbucket.org/fenics-project/dolfin/issues/895.

    if isinstance(u, dolfin.function.function.Function):
        self.read(u._cpp_object)
    else:
        self.read(u)


# Extend cpp.io.File class, and clean-up
cpp.io.File.__lshift__ = __lshift__
cpp.io.File.__rshift__ = __rshift__
del __lshift__, __rshift__
