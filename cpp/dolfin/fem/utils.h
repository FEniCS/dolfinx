// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScMatrix;
class PETScVector;
}
namespace function
{
class Function;
class FunctionSpace;
}

namespace mesh
{
class Mesh;
class MeshGeometry;
}

namespace fem
{
class Form;

/// Initialise matrix. Matrix is not zeroed.
void init(la::PETScMatrix& A, const Form& a);

/// Initialise nested (MatNest) matrix. Matrix is not zeroed.
void init_nest(la::PETScMatrix& A,
               std::vector<std::vector<const fem::Form*>> a);

/// Initialise nested (VecNest) vector. Vector is not zeroed.
void init_nest(la::PETScVector& x, std::vector<const fem::Form*> L);

/// Initialise monolithic  matrix. Matrix is not zeroed.
void init_monolithic(la::PETScMatrix& A,
                     std::vector<std::vector<const fem::Form*>> a);

/// Initialise monolithic vector. Vector is not zeroed.
void init_monolithic(la::PETScVector& x, std::vector<const fem::Form*> L);

/// Initialise vector. Vector is not zeroed.
void init(la::PETScVector& x, const Form& a);

/// Return a map between dof indices and vertex indices
///
/// Only works for function::FunctionSpace with dofs exclusively on vertices.
/// For mixed function::FunctionSpaces vertex index is offset with the number
/// of dofs per vertex.
///
/// In parallel the returned map maps both owned and unowned dofs
/// (using local indices) thus covering all the vertices. Hence the
/// returned map is an inversion of _vertex_to_dof_map_.
///
/// @param    space (_function::FunctionSpace_)
///         The function::FunctionSpace for what the dof to vertex map should
///         be computed for
///
/// @return   std::vector<std::size_t>
///         The dof to vertex map
std::vector<std::size_t>
dof_to_vertex_map(const function::FunctionSpace& space);

/// Return a map between vertex indices and dof indices
///
/// Only works for function::FunctionSpace with dofs exclusively on vertices.
/// For mixed function::FunctionSpaces dof index is offset with the number of
/// dofs per vertex.
///
/// @param    space (_function::FunctionSpace_)
///         The function::FunctionSpace for what the vertex to dof map should
///         be computed for
///
/// @return    std::vector<dolfin::la_index_t>
///         The vertex to dof map
std::vector<dolfin::la_index_t>
vertex_to_dof_map(const function::FunctionSpace& space);

/// Sets mesh coordinates from function
///
/// Mesh connectivities d-0, d-1, ..., d-r are built on function mesh
/// (where d is topological dimension of the mesh and r is maximal
/// dimension of entity associated with any coordinate node). Consider
/// clearing unneeded connectivities when finished.
///
/// @param   geometry (_MeshGeometry_)
///         Mesh geometry to be set
/// @param    position (_Function_)
///         Vectorial Lagrange function with matching degree and mesh
void set_coordinates(mesh::MeshGeometry& geometry,
                     const function::Function& position);

/// Stores mesh coordinates into function
///
/// Mesh connectivities d-0, d-1, ..., d-r are built on function mesh
/// (where d is topological dimension of the mesh and r is maximal
/// dimension of entity associated with any coordinate node). Consider
/// clearing unneeded connectivities when finished.
///
/// @param   position (_Function_)
///         Vectorial Lagrange function with matching degree and mesh
/// @param    geometry (_MeshGeometry_)
///         Mesh geometry to be stored
void get_coordinates(function::Function& position,
                     const mesh::MeshGeometry& geometry);

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
/// @param coordinates
/// (_Function_)
///         Vector Lagrange function of any degree
///
/// @return Mesh
///         The mesh
mesh::Mesh create_mesh(function::Function& coordinates);
}
}
