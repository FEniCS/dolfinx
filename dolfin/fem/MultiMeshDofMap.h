// Copyright (C) 2013-2014 Anders Logg
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
// First added:  2013-09-19
// Last changed: 2014-04-28

#ifndef __MULTI_MESH_DOF_MAP_H
#define __MULTI_MESH_DOF_MAP_H

#include "GenericDofMap.h"

namespace dolfin
{

  // Forward declarations
  class MultiMeshFunctionSpace;

  /// This class handles the mapping of degrees of freedom for MultiMesh
  /// function spaces.

  class MultiMeshDofMap
  {
  public:

    /// Constructor
    MultiMeshDofMap();

    // Copy constructor
    MultiMeshDofMap(const MultiMeshDofMap& dofmap);

    /// Destructor
    ~MultiMeshDofMap();

    /// Return the number dofmaps (parts) of the MultiMesh dofmap
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of dofmaps (parts) of the MultiMesh dofmap
    std::size_t num_parts() const;

    /// Return dofmap (part) number i
    ///
    /// *Returns*
    ///     _GenericDofMap_
    ///         Dofmap (part) number i
    std::shared_ptr<const GenericDofMap> part(std::size_t i) const;

    /// Add dofmap (shared pointer version)
    ///
    /// *Arguments*
    ///     dofmap (_GenericDofMap_)
    ///         The dofmap.
    void add(std::shared_ptr<const GenericDofMap> dofmap);

    /// Add dofmap (reference version)
    ///
    /// *Arguments*
    ///     dofmap (_DofMap_)
    ///         The dofmap.
    void add(const GenericDofMap& dofmap);

    /// Build MultiMesh dofmap
    void build(const MultiMeshFunctionSpace& function_space);

    /// Clear MultiMesh dofmap
    void clear();

    /// Return the dimension of the global finite element function
    /// space
    std::size_t global_dimension() const;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    std::pair<std::size_t, std::size_t> ownership_range() const;

    /// Return map from nonlocal-dofs (that appear in local dof map)
    /// to owning process
    const boost::unordered_map<std::size_t,
                               unsigned int>& off_process_owner() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Total global dimension (sum of parts)
    std::size_t _global_dimension;

    // List of original dofmaps
    std::vector<std::shared_ptr<const GenericDofMap> > _dofmaps;

    // List of dofmaps
    std::vector<std::shared_ptr<GenericDofMap> > _new_dofmaps;

    // Local-to-global dof map for all parts
    std::vector<std::vector<std::vector<dolfin::la_index> > > _dofmap;


  };

}

#endif
