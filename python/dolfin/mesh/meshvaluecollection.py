
import dolfin.cpp as cpp

_meshvaluecollection_types = {"bool": cpp.mesh.MeshValueCollection_bool,
                              "size_t": cpp.mesh.MeshValueCollection_sizet,
                              "int": cpp.mesh.MeshValueCollection_int,
                              "double": cpp.mesh.MeshValueCollection_double}


class MeshValueCollection(object):
    def __new__(cls, value_type, mesh, dim=None):
        if value_type not in _meshvaluecollection_types.keys():
            raise KeyError("MeshValueCollection type not recognised")
        mvc = _meshvaluecollection_types[value_type]
        if dim is not None:
            return mvc(mesh, dim)
        else:
            return mvc(mesh)
