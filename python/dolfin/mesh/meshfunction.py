
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
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        if value is not None:
            return fn(mesh, 0, value)
        else:
            return fn(mesh, 0)


class EdgeFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        if value is not None:
            return fn(mesh, 1, value)
        else:
            return fn(mesh, 1)


class FaceFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        if value is not None:
            return fn(mesh, 2, value)
        else:
            return fn(mesh, 2)


class FacetFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        tdim = mesh.topology().dim() - 1
        if value is not None:
            return fn(mesh, tdim, value)
        else:
            return fn(mesh, tdim)


class CellFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        tdim = mesh.topology().dim()
        if value is not None:
            return fn(mesh, tdim, value)
        else:
            return fn(mesh, tdim)
