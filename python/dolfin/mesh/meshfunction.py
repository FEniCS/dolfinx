
import dolfin.cpp as cpp

_meshfunction_types = {"bool": cpp.mesh.MeshFunctionBool,
                       "size_t": cpp.mesh.MeshFunctionSizet,
                       "int": cpp.mesh.MeshFunctionInt,
                       "double": cpp.mesh.MeshFunctionDouble}

_vertexfunction_types = {"bool": cpp.mesh.VertexFunctionBool,
                       "size_t": cpp.mesh.VertexFunctionSizet,
                       "int": cpp.mesh.VertexFunctionInt,
                       "double": cpp.mesh.VertexFunctionDouble}

_edgefunction_types = {"bool": cpp.mesh.EdgeFunctionBool,
                       "size_t": cpp.mesh.EdgeFunctionSizet,
                       "int": cpp.mesh.EdgeFunctionInt,
                       "double": cpp.mesh.EdgeFunctionDouble}

_facefunction_types = {"bool": cpp.mesh.FaceFunctionBool,
                       "size_t": cpp.mesh.FaceFunctionSizet,
                       "int": cpp.mesh.FaceFunctionInt,
                       "double": cpp.mesh.FaceFunctionDouble}

_facefunction_types = {"bool": cpp.mesh.FaceFunctionBool,
                       "size_t": cpp.mesh.FaceFunctionSizet,
                       "int": cpp.mesh.FaceFunctionInt,
                       "double": cpp.mesh.FaceFunctionDouble}

_facetfunction_types = {"bool": cpp.mesh.FacetFunctionBool,
                       "size_t": cpp.mesh.FacetFunctionSizet,
                       "int": cpp.mesh.FacetFunctionInt,
                       "double": cpp.mesh.FacetFunctionDouble}

_cellfunction_types = {"bool": cpp.mesh.CellFunctionBool,
                       "size_t": cpp.mesh.CellFunctionSizet,
                       "int": cpp.mesh.CellFunctionInt,
                       "double": cpp.mesh.CellFunctionDouble}



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
        fn = _vertexfunction_types[value_type]
        if value is not None:
            return fn(mesh, value)
        else:
            return fn(mesh)


class EdgeFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _edgefunction_types[value_type]
        if value is not None:
            return fn(mesh, value)
        else:
            return fn(mesh)


class FaceFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _facefunction_types[value_type]
        if value is not None:
            return fn(mesh, value)
        else:
            return fn(mesh)


class FacetFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _facetfunction_types[value_type]
        if value is not None:
            return fn(mesh, value)
        else:
            return fn(mesh)


class CellFunction(object):
    def __new__(cls, value_type, mesh, value=None):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _cellfunction_types[value_type]
        if value is not None:
            return fn(mesh, value)
        else:
            return fn(mesh)
