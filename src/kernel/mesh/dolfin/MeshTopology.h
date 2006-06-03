// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-06-03

#ifndef __MESH_TOPOLOGY_H
#define __MESH_TOPOLOGY_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/MeshConnectivity.h>

namespace dolfin
{
  
  /// MeshTopology stores the topology of a mesh, consisting of mesh entities
  /// and connectivity (incidence relations for the mesh entities). Note that
  /// the mesh entities don't need to be stored, only the number of entities
  /// and the connectivity. Any numbering scheme for the mesh entities is
  /// stored separately in a MeshFunction over the entities.
  ///
  /// A mesh entity may be identified globally as a pair (d, e), where d is
  /// the topological dimension and e is the number of the entity within
  /// that topological dimension.
  
  class MeshTopology
  {
  public:
    
    /// Create empty mesh topology
    MeshTopology();
    
    /// Destructor
    ~MeshTopology();
 
    /// Return topological dimension
    inline uint dim() const { return _dim; }
    
    /// Return number of entities for given dimension
    inline uint size(uint dim) const
    { dolfin_assert(dim <= _dim); return num_entities[dim]; }

    /// Clear all data
    void clear();

    /// Initialize topology of given maximum dimension
    void init(uint dim);

    /// Set number of entities (size) for given topological dimension
    void init(uint dim, uint size);

    /// Return connectivity for given pair of topological dimensions
    inline MeshConnectivity& operator() (uint d0, uint d1)
    { dolfin_assert(d0 <= _dim && d1 <= _dim); return connectivity[d0][d1]; }

    /// Return connectivity for given pair of topological dimensions
    inline const MeshConnectivity& operator() (uint d0, uint d1) const
    { dolfin_assert(d0 <= _dim && d1 <= _dim); return connectivity[d0][d1]; }

    /// Display data
    void disp() const;

  private:

    // Topological dimension
    uint _dim;
  
    // Number of mesh entities for each topological dimension
    uint* num_entities;

    // Connectivity for pairs of topological dimensions
    MeshConnectivity** connectivity;
   
  };

}

#endif
