// Copyright (C) 2013 Johan Hake
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
//
// First added:  2013-09-05
// Last changed: 2013-09-05

#ifndef __FEM_UTILS_H
#define __FEM_UTILS_H

#include <vector>

#include <dolfin/common/types.h>

namespace dolfin
{

  class FunctionSpace;

  /// Return a map between dofs and vertex indices
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh to create the map between
  ///
  /// *Returns*
  ///     std::vector<dolfin::la_index>
  ///         The dof to vertex map
  std::vector<dolfin::la_index> dof_to_vertex_map(const FunctionSpace& space);

  /// Return a map between vertex indices and dofs
  ///
  /// *Arguments*
  ///     mesh (_Mesh_)
  ///         The mesh to create the map between
  ///
  /// *Returns*
  ///     std::vector<std::size_t>
  ///         The vertex to dof map
  std::vector<std::size_t> vertex_to_dof_map(const FunctionSpace& space);

}

#endif
