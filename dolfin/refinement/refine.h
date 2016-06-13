// Copyright (C) 2010 Garth N. Wells
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
// Modified by Anders Logg, 2010.
//
// First added:  2010-02-10
// Last changed: 2013-05-12
//
// This file defines free functions for mesh refinement.
//

#ifndef __DOLFIN_REFINE_H
#define __DOLFIN_REFINE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class MeshHierarchy;
  template <typename T> class MeshFunction;

  /// Create uniformly refined mesh
  ///
  /// @param    mesh (_Mesh_)
  ///         The mesh to refine.
  /// @param    redistribute (_bool_)
  ///         Optional argument to redistribute the refined mesh if mesh is a
  ///         distributed mesh.
  ///
  /// @return    _Mesh_
  ///         The refined mesh.
  ///
  /// @code{.cpp}
  ///         mesh = refine(mesh);
  /// @endcode
  ///
  Mesh refine(const Mesh& mesh, bool redistribute = true);

  /// Refine a MeshHierarchy
  std::shared_ptr<const MeshHierarchy> refine(
              const MeshHierarchy& hierarchy,
              const MeshFunction<bool>& markers);

  /// Create uniformly refined mesh
  ///
  /// @param    refined_mesh (_Mesh_)
  ///         The mesh that will be the refined mesh.
  /// @param   mesh (_Mesh_)
  ///         The original mesh.
  /// @param    redistribute (_bool_)
  ///         Optional argument to redistribute the refined mesh if mesh is a
  ///         distributed mesh.
  void refine(Mesh& refined_mesh, const Mesh& mesh,
              bool redistribute = true);

  /// Create locally refined mesh
  ///
  /// @param  mesh (_Mesh_)
  ///         The mesh to refine.
  /// @param cell_markers (_MeshFunction<bool>_)
  ///         A mesh function over booleans specifying which cells
  ///         that should be refined (and which should not).
  /// @param redistribute (_bool_)
  ///         Optional argument to redistribute the refined mesh if mesh is a
  ///         distributed mesh.
  ///
  /// @return _Mesh_
  ///         The locally refined mesh.
  ///
  /// @code{.cpp}
  ///         CellFunction<bool> cell_markers(mesh);
  ///         cell_markers.set_all(false);
  ///         Point origin(0.0, 0.0, 0.0);
  ///         for (CellIterator cell(mesh); !cell.end(); ++cell)
  ///         {
  ///             Point p = cell->midpoint();
  ///             if (p.distance(origin) < 0.1)
  ///                 cell_markers[*cell] = true;
  ///         }
  ///         mesh = refine(mesh, cell_markers);
  /// @endcode
  ///
  Mesh refine(const Mesh& mesh, const MeshFunction<bool>& cell_markers,
              bool redistribute = true);

  /// Create locally refined mesh
  ///
  /// @param    refined_mesh (_Mesh_)
  ///         The mesh that will be the refined mesh.
  /// @param    mesh (_Mesh_)
  ///         The original mesh.
  /// @param    cell_markers (_MeshFunction<bool>_)
  ///         A mesh function over booleans specifying which cells
  ///         that should be refined (and which should not).
  /// @param    redistribute (_bool_)
  ///         Optional argument to redistribute the refined mesh if mesh is a
  ///         distributed mesh.
  void refine(Mesh& refined_mesh, const Mesh& mesh,
              const MeshFunction<bool>& cell_markers, bool redistribute = true);

  /// Increase the polynomial order of the mesh from 1 to 2, i.e. add points
  /// at the Edge midpoints, to make a quadratic mesh.
  ///
  /// @param    refined_mesh (_Mesh_)
  ///         The mesh that will be the quadratic mesh.
  /// @param    mesh (_Mesh_)
  ///         The original linear mesh.
  void p_refine(Mesh& refined_mesh, const Mesh& mesh);

  /// Return a p_refined mesh
  /// Increase the polynomial order of the mesh from 1 to 2, i.e. add points
  /// at the Edge midpoints, to make a quadratic mesh.
  ///
  ///  @param   mesh (_Mesh_)
  ///         The original linear mesh.
  ///  @return _Mesh_
  Mesh p_refine(const Mesh& mesh);
}

#endif
