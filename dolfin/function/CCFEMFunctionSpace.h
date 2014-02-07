// Copyright (C) 2013 Anders Logg
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
// First added:  2013-08-05
// Last changed: 2014-02-07

#ifndef __CCFEM_FUNCTION_SPACE_H
#define __CCFEM_FUNCTION_SPACE_H

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  // Forward declarations
  class FunctionSpace;
  class CCFEMDofMap;
  class Mesh;
  class BoundingBoxTree;
  class BoundaryMesh;

  /// This class represents a cut and composite finite element
  /// function space (CCFEM) defined on one or more possibly
  /// intersecting meshes.
  ///
  /// FIXME: Document usage of class with add() followed by build()

  class CCFEMFunctionSpace
  {
  public:

    /// Create empty CCFEM function space
    CCFEMFunctionSpace();

    /// Destructor
    ~CCFEMFunctionSpace();

    /// Return dimension of the CCFEM function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension of the CCFEM function space.
    std::size_t dim() const;

    /// Return CCFEM dofmap
    ///
    /// *Returns*
    ///     _CCFEMDofMap_
    ///         The dofmap.
    boost::shared_ptr<const CCFEMDofMap> dofmap() const;

    /// Return the number function spaces (parts) of the CCFEM function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of function spaces (parts) of the CCFEM function space.
    std::size_t num_parts() const;

    /// Return function space (part) number i
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         Function space (part) number i
    boost::shared_ptr<const FunctionSpace> part(std::size_t i) const;

    /// Add function space (shared pointer version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(boost::shared_ptr<const FunctionSpace> function_space);

    /// Add function space (reference version)
    ///
    /// *Arguments*
    ///     function_space (_FunctionSpace_)
    ///         The function space.
    void add(const FunctionSpace& function_space);

    /// Build CCFEM function space
    void build();

    /// Clear CCFEM function space
    void clear();

  private:

    // List of function spaces
    std::vector<boost::shared_ptr<const FunctionSpace> > _function_spaces;

    // CCFEM dofmap
    boost::shared_ptr<CCFEMDofMap> _dofmap;

    // List of meshes
    std::vector<boost::shared_ptr<const Mesh> > _meshes;

    // List of boundary meshes
    std::vector<boost::shared_ptr<BoundaryMesh> > _boundary_meshes;

    // List of bounding box trees for meshes
    std::vector<boost::shared_ptr<BoundingBoxTree> > _trees;

    // List of bounding box trees for boundary meshes
    std::vector<boost::shared_ptr<BoundingBoxTree> > _boundary_trees;

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

    // Collision map for cut cells. Access data by
    //
    // where
    //
    //     c.first  = part number for the cutting mesh
    //     c.second = cell index for the cutting cell
    //            i = the part (mesh) number
    //            j = the cell number (in the list of cut cells)
    //            k = the collision number (in the list of cutting cells)
    std::vector<std::map<unsigned int,
                         std::vector<std::pair<std::size_t, unsigned int> > > >
    _collision_map_cut_cells;

    // Build dofmap
    void _build_dofmap();

    // Build boundary meshes
    void _build_boundary_meshes();

    // Build bounding box trees
    void _build_bounding_box_trees();

    // Build collision maps
    void _build_collision_maps();

  };

}

#endif
