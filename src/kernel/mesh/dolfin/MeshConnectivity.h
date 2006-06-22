// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-06-22

#ifndef __MESH_CONNECTIVITY_H
#define __MESH_CONNECTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

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

    /// Create empty connectivity
    MeshConnectivity();

    /// Copy constructor
    MeshConnectivity(const MeshConnectivity& connectivity);

    /// Destructor
    ~MeshConnectivity();

    /// Assignment
    const MeshConnectivity& operator= (const MeshConnectivity& connectivity);

    /// Return total number of connections
    inline uint size() const { return _size; }

    /// Return number of connections for given entity
    inline uint size(uint entity) const
    { dolfin_assert(entity < num_entities); return offsets[entity + 1] - offsets[entity]; }

    /// Return array of connections for given entity
    inline uint* operator() (uint entity)
    { dolfin_assert(entity < num_entities); return connections + offsets[entity]; }

    /// Clear all data
    void clear();

    /// Initialize number of entities and number of connections (equal for all)
    void init(uint num_entities, uint num_connections);

    /// Initialize number of entities and number of connections (individually)
    void init(Array<uint>& num_connections);

    /// Set given connection for given entity
    void set(uint entity, uint connection, uint pos);

    /// Set all connections for given entity
    void set(uint entity, const Array<uint>& connections);

    /// Set all connections for given entity
    void set(uint entity, uint* connections);

    /// Set all connections for all entities
    void set(const Array<Array<uint> >& connectivity);

    /// Display data
    void disp() const;
    
  private:

    /// Total number of connections
    uint _size;

    /// Number of entities
    uint num_entities;
    
    /// Connections for all entities stored as a contiguous array
    uint* connections;

    /// Offset for first connection for each entity
    uint* offsets;

  };

}

#endif
