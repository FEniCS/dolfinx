// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{

namespace mesh
{
// Forward declarations
class Mesh;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace refinement
{

/// Create uniformly refined mesh
///
/// @param    mesh (_mesh::Mesh_)
///         The mesh to refine.
/// @param    redistribute (_bool_)
///         Optional argument to redistribute the refined mesh if mesh is a
///         distributed mesh.
///
/// @return    _mesh::Mesh_
///         The refined mesh.
///
/// @code{.cpp}
///         mesh = refine(mesh);
/// @endcode
///
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute = true);

/// Create locally refined mesh
///
/// @param  mesh (_mesh::Mesh_)
///         The mesh to refine.
/// @param cell_markers (_mesh::MeshFunction<int>_)
///         A mesh function over integers specifying which cells
///         should be refined (value == 1) (and which should not
///         (any other integer value)).
/// @param redistribute (_bool_)
///         Optional argument to redistribute the refined mesh if mesh is a
///         distributed mesh.
///
/// @return _mesh::Mesh_
///         The locally refined mesh.
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const mesh::MeshFunction<int>& cell_markers,
                  bool redistribute = true);

} // namespace refinement
} // namespace dolfin
