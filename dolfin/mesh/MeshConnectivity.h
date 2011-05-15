// Copyright (C) 2006-2007 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-05-09
// Last changed: 2010-11-28

#ifndef __MESH_CONNECTIVITY_H
#define __MESH_CONNECTIVITY_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// Mesh connectivity stores a sparse data structure of connections
  /// (incidence relations) between mesh entities for a fixed pair of
  /// topological dimensions.
  ///
  /// The connectivity can be specified either by first giving the
  /// number of entities and the number of connections for each entity,
  /// which may either be equal for all entities or different, or by
  /// giving the entire (sparse) connectivity pattern.

  class MeshConnectivity
  {
  public:

    /// Create empty connectivity between given dimensions (d0 -- d1)
    MeshConnectivity(uint d0, uint d1);

    /// Copy constructor
    MeshConnectivity(const MeshConnectivity& connectivity);

    /// Destructor
    ~MeshConnectivity();

    /// Assignment
    const MeshConnectivity& operator= (const MeshConnectivity& connectivity);

    /// Return total number of connections
    uint size() const { return _size; }

    /// Return number of connections for given entity
    uint size(uint entity) const
    { return (entity < num_entities ? offsets[entity + 1] - offsets[entity] : 0); }

    /// Return array of connections for given entity
    const uint* operator() (uint entity) const
    { return (entity < num_entities ? connections + offsets[entity] : 0); }

    /// Return contiguous array of connections for all entities
    const uint* operator() () const { return connections; }

    /// Clear all data
    void clear();

    /// Initialize number of entities and number of connections (equal for all)
    void init(uint num_entities, uint num_connections);

    /// Initialize number of entities and number of connections (individually)
    void init(std::vector<uint>& num_connections);

    /// Set given connection for given entity
    void set(uint entity, uint connection, uint pos);

    /// Set all connections for given entity
    void set(uint entity, const std::vector<uint>& connections);

    /// Set all connections for given entity
    void set(uint entity, uint* connections);

    /// Set all connections for all entities
    void set(const std::vector<std::vector<uint> >& connectivity);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class BinaryFile;
    friend class MeshRenumbering;

    // Dimensions (only used for pretty-printing)
    uint d0, d1;

    // Total number of connections
    uint _size;

    // Number of entities
    uint num_entities;

    // Connections for all entities stored as a contiguous array
    uint* connections;

    // Offset for first connection for each entity
    uint* offsets;

  };

}

#endif
