import sys
import dolfin.cpp as cpp

_meshfunction_types = {"bool": cpp.mesh.MeshFunctionBool,
                       "size_t": cpp.mesh.MeshFunctionSizet,
                       "int": cpp.mesh.MeshFunctionInt,
                       "double": cpp.mesh.MeshFunctionDouble}

class MeshFunction(object):
    def __new__(cls, value_type, mesh, dim, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        if value is not None:
            return fn(mesh, dim, value)
        else:
            return fn(mesh, dim)


class VertexFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        print('-------------------------------------------\n')
        print('VertexFunction is deprecated since 2017.2.0')
        print('Use MeshFunction<T>(mesh, 0) instead)
        print('-------------------------------------------')
        sys.exit()


class EdgeFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        print('-------------------------------------------\n')
        print('EdgeFunction is deprecated since 2017.2.0')
        print('Use MeshFunction<T>(mesh, 1) instead)
        print('-------------------------------------------')
        sys.exit()


class FaceFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        print('-------------------------------------------\n')
        print('FaceFunction is deprecated since 2017.2.0')
        print('Use MeshFunction<T>(mesh, 2) instead)
        print('-------------------------------------------')
        sys.exit()


class FacetFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        print('-------------------------------------------\n')
        print('FacetFunction is deprecated since 2017.2.0')
        print('Use MeshFunction<T>(mesh, mesh.topology().dim()-1) instead)
        print('-------------------------------------------')
        sys.exit()


class CellFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        print('-------------------------------------------\n')
        print('CellFunction is deprecated since 2017.2.0')
        print('Use MeshFunction<T>(mesh, mesh.topology().dim()) instead)
        print('-------------------------------------------')
        sys.exit()
