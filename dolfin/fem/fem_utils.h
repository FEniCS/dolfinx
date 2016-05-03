// Copyright (C) 2013, 2015 Johan Hake, Jan Blechta
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

  /// Return a map between dofs indices and vertex indices
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
  ///         The FunctionSpace for what the dof to vertex map should
  ///         be computed for
  ///
  /// *Returns*
  ///     std::vector<std::size_t>
  ///         The dof to vertex map
  std::vector<std::size_t> dof_to_vertex_map(const FunctionSpace& space);

  /// Return a map between vertex indices and dofs indices
  ///
  /// Only works for FunctionSpace with dofs exclusively on vertices.
  /// For mixed FunctionSpaces dof index is offset with the number of
  /// dofs per vertex.
  ///
  /// *Arguments*
  ///     space (_FunctionSpace_)
  ///         The FunctionSpace for what the vertex to dof map should
  ///         be computed for
  ///
  /// *Returns*
  ///     std::vector<dolfin::la_index>
  ///         The vertex to dof map
  std::vector<dolfin::la_index> vertex_to_dof_map(const FunctionSpace& space);

}

#endif
