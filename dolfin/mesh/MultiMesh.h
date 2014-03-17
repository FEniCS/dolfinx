// Copyright (C) 2014 Anders Logg
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
// First added:  2014-03-03
// Last changed: 2014-03-14

#ifndef __MULTI_MESH_H
#define __MULTI_MESH_H

#include <vector>
#include <map>

namespace dolfin
{

  // Forward declarations
  class Cell;
  class Mesh;
  class BoundaryMesh;
  class BoundingBoxTree;

  /// This class represents a collection of meshes with arbitrary
  /// overlaps. A multimesh may be created from a set of standard
  /// meshes spaces by repeatedly calling add(), followed by a call to
  /// build(). Note that a multimesh is not useful until build() has
  /// been called.

  class MultiMesh
  {
  public:

    /// Create empty multimesh
    MultiMesh();

    /// Destructor
    ~MultiMesh();

    /// Return the number of meshes (parts) of the multimesh
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of meshes (parts) of the multimesh.
    std::size_t num_parts() const;

    /// Return mesh (part) number i
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     _Mesh_
    ///         Mesh (part) number i
    std::shared_ptr<const Mesh> part(std::size_t i) const;

    /// Return the list of uncut cells for given part. The uncut cells
    /// are defined as all cells that don't collide with any cells in
    /// any other part with higher part number.
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of uncut cell indices for given part
    const std::vector<unsigned int>& uncut_cells(std::size_t part) const;

    /// Return the list of cut cells for given part. The cut cells are
    /// defined as all cells that collide with the boundary of any
    /// part with higher part number.
    ///
    /// FIXME: Figure out whether this makes sense; a cell may collide
    /// with the boundary of part j but may still be covered
    /// completely by the domain of part j + 1. Possible solution is
    /// to for each part i check overlapping parts starting from the
    /// top and working back down to i + 1.
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of cut cell indices for given part
    const std::vector<unsigned int>& cut_cells(std::size_t part) const;

    /// Return the list of covered cells for given part. The covered
    /// cells are defined as all cells that collide with the domain of
    /// any part with higher part number, but not with the boundary of
    /// that part; in other words cells that are completely covered by
    /// any other part (and which therefore are inactive).
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         List of covered cell indices for given part
    const std::vector<unsigned int>& covered_cells(std::size_t part) const;

    /// Return the collision map for cut cells of the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::vector<std::pair<std::size_t, unsigned int> > >
    ///         A map from cell indices of cut cells to a list of
    ///         cutting cells. Each cutting cell is represented as a
    ///         pair (part_number, cutting_cell_index).
    const std::map<unsigned int,
                   std::vector<std::pair<std::size_t, unsigned int> > >&
    collision_map_cut_cells(std::size_t part) const;

    /// Return quadrature rules for cut cells of the given part
    ///
    /// *Arguments*
    ///     part (std::size_t)
    ///         The part number
    ///
    /// *Returns*
    ///     std::map<unsigned int, std::pair<std::vector<double>, std::vector<double> > >
    ///         A map from cell indices of cut cells to a quadrature
    ///         rules. Each quadrature rule is represented as a pair
    ///         of an array of quadrature weights and a corresponding
    ///         flattened array of quadrature points.
    const std::map<unsigned int, std::pair<std::vector<double>, std::vector<double> > > &
    quadrature_rule_cut_cells(std::size_t part) const;

    /// Add mesh (shared pointer version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    void add(std::shared_ptr<const Mesh> mesh);

    /// Add mesh (reference version)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh
    void add(const Mesh& mesh);

    /// Build multimesh
    void build();

    /// Clear multimesh
    void clear();

  private:

    // List of meshes
    std::vector<std::shared_ptr<const Mesh> > _meshes;

    // List of boundary meshes
    std::vector<std::shared_ptr<BoundaryMesh> > _boundary_meshes;

    // List of bounding box trees for meshes
    std::vector<std::shared_ptr<BoundingBoxTree> > _trees;

    // List of bounding box trees for boundary meshes
    std::vector<std::shared_ptr<BoundingBoxTree> > _boundary_trees;

    // Cell indices for all uncut cells for all parts. Access data by
    //
    //     c = _uncut_cells[i][j]
    //
    // where
    //
    //     c = cell index for an uncut cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of uncut cells)
    std::vector<std::vector<unsigned int> > _uncut_cells;

    // Cell indices for all cut cells for all parts. Access data by
    //
    //     c = _cut_cells[i][j]
    //
    // where
    //
    //     c = cell index for a cut cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of cut cells)
    std::vector<std::vector<unsigned int> > _cut_cells;

    // Cell indices for all covered cells for all parts. Access data by
    //
    //     c = _covered_cells[i][j]
    //
    // where
    //
    //     c = cell index for a covered cell
    //     i = the part (mesh) number
    //     j = the cell number (in the list of covered cells)
    std::vector<std::vector<unsigned int> > _covered_cells;

    // Developer note 1: The data structures _collision_map_cut_cells
    // and _quadrature_rules_cut_cells may be changed from maps to
    // vectors and indexed by the number of the cut cell (in the list
    // of cut cells), instead of indexed by the local cell index as
    // here, if we find that this is important for performance.
    //
    // Developer note 2: Quadrature points are naturally a part of a
    // form (or a term in a form) and not a part of a mesh. However,
    // for now we use a global (to the multimesh) quadrature rule for
    // all cut cells, for simplicity.

    // Collision map for cut cells. Access data by
    //
    //     c = _collision_map_cut_cells[i][j][k]
    //
    // where
    //
    //     c.first  = part number for the cutting mesh
    //     c.second = cell index for the cutting cell
    //            i = the part (mesh) number
    //            j = the cell number (local cell index)
    //            k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int,
                         std::vector<std::pair<std::size_t, unsigned int> > > >
    _collision_maps_cut_cells;

    // Quadrature rules for cut cells. Access data by
    //
    //     q = _quadrature_rules_cut_cells[i][j]
    //
    // where
    //
    //     q.first  = quadrature weights, array of length num_points
    //     q.second = quadrature points, flattened num_points x gdim array
    //            i = the part (mesh) number
    //            j = the cell number (local cell index)
    std::vector<std::map<unsigned int,
                         std::pair<std::vector<double>, std::vector<double> > > >
    _quadrature_rules_cut_cells;

    // Build boundary meshes
    void _build_boundary_meshes();

    // Build bounding box trees
    void _build_bounding_box_trees();

    // Build collision maps
    void _build_collision_maps();

    // Build quadrature rules
    void _build_quadrature_rules();

    // Add quadrature rule for intersection between cell_0 and cell_1
    void _add_quadrature_rule(std::pair<std::vector<double>,
                                        std::vector<double> >& quadrature_rule,
                              const Cell& cell_0,
                              const Cell& cell_1,
                              std::size_t tdim,
                              std::size_t gdim,
                              std::size_t order,
                              double factor) const;

  };

}

#endif
