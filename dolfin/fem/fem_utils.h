// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef __FEM_UTILS_H
#define __FEM_UTILS_H

#include <vector>

#include <dolfin/common/types.h>

namespace dolfin
{

  class FunctionSpace;


  /// Return a map between entities marked in a meshfunction and dof indices
  ///
  /// *Arguments*
  ///     num_marked_entities (std::size_t& )
  ///         Return value: the number of entities in subdomains marked with subdomain_id
  ///     marked_to_mesh_indices (std::vector<std::size_t>&)
  ///         Return value: the mesh index of each marked entity
  ///     entity_to_dofs (std::vector<dolfin::la_index>&)
  ///         Return value: The set of dofs associated with marked entities,
  ///         with array dimension num_marked_entities * dofs_per_entity
  ///     space (_FunctionSpace_)
  ///         The FunctionSpace for which the entity to
  ///         dof map should be computed for
  ///     subdomains (_MeshFunction_)
  ///         Subdomain markers of any entity dimension, on the
  ///         same mesh as the function space
  ///     subdomain_id (std::size_t)
  ///         Marker value to select entities from subdomain markers
  /// *Returns*
  ///     num_marked_entities, marked_to_mesh_entities, entity_to_dofs
  /*
  void entity_to_dof_map(std::size_t& num_marked_entities,
     std::vector<std::size_t>& marked_to_mesh_indices,
     std::vector<dolfin::la_index>& entity_to_dofs,
     const FunctionSpace& space,
     const MeshFunction<std::size_t>& subdomains,
     std::size_t subdomain_id);
  */
/*
  // Usage:
  int num_dofs_per_entity = entity_to_dofs.size() / marked_to_mesh_indices.size();
  for (int i=0; i < marked_to_mesh_indices.size(); ++i)
  {
    int j = marked_to_mesh_indices[i];
    for (int k=0; k < num_dofs_per_entity; ++k)
      int dof = entity_to_dofs[i * num_dofs_per_entity + k];
  }
*/
//f.vector()[entitiy_to_dofs] = g.vector()[entitiy_to_dofs]

  std::vector<dolfin::la_index>
    aggregate_entity_dofs(const FunctionSpace& space,
                      std::size_t entity_dim,
                      const std::vector<std::size_t> & entity_indices);

  std::vector<dolfin::la_index>
    aggregate_entity_dofs(const FunctionSpace& space,
                      std::size_t entity_dim);

  std::vector<dolfin::la_index>
    aggregate_subcomplex_dofs(const FunctionSpace& space,
                      std::size_t entity_dim,
                      const std::vector<std::size_t> & entity_indices);

  std::vector<dolfin::la_index>
    aggregate_subcomplex_dofs(const FunctionSpace& space,
                      std::size_t entity_dim);

  /// Return a map between dof indices and vertex indices
  ///
  /// Only works for FunctionSpace with dofs exclusively on vertices.
  /// For mixed FunctionSpaces vertex index is offset with the number
  /// of dofs per vertex.
  ///
  /// In parallel the returned map maps both owned and unowned dofs
  /// (using local indices) thus covering all the vertices. Hence the
  /// returned map is an inversion of _vertex_to_dof_map_.
  ///
  /// *Arguments*
  ///     space (_FunctionSpace_)
  ///         The FunctionSpace for which the dof to vertex map should
  ///         be computed for
  ///
  /// *Returns*
  ///     std::vector<std::size_t>
  ///         The dof to vertex map
  std::vector<std::size_t> dof_to_vertex_map(const FunctionSpace& space);

  /// Return a map between vertex indices and dof indices
  ///
  /// Only works for FunctionSpace with dofs exclusively on vertices.
  /// For mixed FunctionSpaces dof index is offset with the number of
  /// dofs per vertex.
  ///
  /// *Arguments*
  ///     space (_FunctionSpace_)
  ///         The FunctionSpace for which the vertex to dof map should
  ///         be computed for
  ///
  /// *Returns*
  ///     std::vector<dolfin::la_index>
  ///         The vertex to dof map
  std::vector<dolfin::la_index> vertex_to_dof_map(const FunctionSpace& space);

  class Function;
  class MeshGeometry;

  /// Sets mesh coordinates from function
  ///
  /// Mesh connectivities d-0, d-1, ..., d-r are built on function mesh
  /// (where d is topological dimension of the mesh and r is maximal
  /// dimension of entity associated with any coordinate node). Consider
  /// clearing unneeded connectivities when finished.
  ///
  /// *Arguments*
  ///     geometry (_MeshGeometry_)
  ///         Mesh geometry to be set
  ///     position (_Function_)
  ///         Vectorial Lagrange function with matching degree and mesh
  void set_coordinates(MeshGeometry& geometry, const Function& position);

  /// Stores mesh coordinates into function
  ///
  /// Mesh connectivities d-0, d-1, ..., d-r are built on function mesh
  /// (where d is topological dimension of the mesh and r is maximal
  /// dimension of entity associated with any coordinate node). Consider
  /// clearing unneeded connectivities when finished.
  ///
  /// *Arguments*
  ///     position (_Function_)
  ///         Vectorial Lagrange function with matching degree and mesh
  ///     geometry (_MeshGeometry_)
  ///         Mesh geometry to be stored
  void get_coordinates(Function& position, const MeshGeometry& geometry);

  class Mesh;

  /// Creates mesh from coordinate function
  ///
  /// Topology is given by underlying mesh of the function space and
  /// geometry is given by function values. Hence resulting mesh
  /// geometry has a degree of the function space degree. Geometry of
  /// function mesh is ignored.
  ///
  /// Mesh connectivities d-0, d-1, ..., d-r are built on function mesh
  /// (where d is topological dimension of the mesh and r is maximal
  /// dimension of entity associated with any coordinate node). Consider
  /// clearing unneeded connectivities when finished.
  ///
  /// *Arguments*
  ///     position (_Function_)
  ///         Vectorial Lagrange function with of any degree
  ///
  /// *Returns*
  ///     _Mesh_
  ///         The mesh
  Mesh create_mesh(Function& position);
}

#endif
