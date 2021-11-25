# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Graph module"""

# import typing


from dolfinx import cpp as _cpp


# def _create_adjacencylist(obj):
#     class AdjacencyList(obj.__class__):
#         pass
#     obj.__class__ = AdjacencyList


# class AdjacencyList:
#     pass

# class AdjacencyList:
#     def __init__(self, obj: typing.Union[_cpp.graph.AdjacencyList_int32,
#                                          _cpp.graph.AdjacencyList_int64]):
#         self._cpp_object = obj

#     @classmethod
#     def from_data(cls, data, offsets):
#         """Create an AdjacencyList from the adjacency data and an array
#         of offsets in the data. The AdjacencyList copies the data and
#         offset arrays."""
#         if data.dtype == np.int64:
#             cpp_object = _cpp.graph.AdjacencyList_int64(data, offsets)
#         elif data.dtype == np.int32:
#             cpp_object = _cpp.graph.AdjacencyList_int32(data, offsets)
#         else:
#             raise RuntimeError("Unsupported dtype")
#         return cls(cpp_object)

#     @classmethod
#     def from_array(cls, data):
#         """Create an AdjacencyList from a rectangular array. The degree
#         of each node is the same and is equal ot the number of columns
#         in data."""
#         if data.dtype == np.int64:
#             cpp_object = _cpp.graph.AdjacencyList_int64(data)
#         elif data.dtype == np.int32:
#             cpp_object = _cpp.graph.AdjacencyList_int32(data)
#         else:
#             raise RuntimeError("Unsupported dtype")
#         return cls(cpp_object)

#     def links(self, i):
#         """The links (edges) from the ith node"""
#         return self._cpp_object.links(i)

#     @property
#     def array(self):
#         """Array holding the adjacncy list links (edges)"""
#         return self._cpp_object.array

#     @property
#     def offsets(self):
#         """Offset into array for the links from the ith node"""
#         return self._cpp_object.offsets

#     def __eq__(self, other):
#         return self._cpp_object == other._cpp_object

#     def __repr__(self):
#         return repr(self._cpp_object)

#     def __len__(self):
#         """Number of nodes in the adjacencylist"""
#         return len(self._cpp_object)


def create_adjacencylist(data, offsets=None):
    """Create an AdjacencyList"""
    if offsets is None:
        try:
            return _cpp.graph.AdjacencyList_int32(data)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data)
    else:
        try:
            return _cpp.graph.AdjacencyList_int32(data, offsets)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data, offsets)
