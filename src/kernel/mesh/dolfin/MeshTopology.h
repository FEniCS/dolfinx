// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-05-22

#ifndef __MESH_TOPOLOGY_H
#define __MESH_TOPOLOGY_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin
{

  class MeshConnectivity;
  
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
    inline uint size(uint dim) const { dolfin_assert(dim < _dim); return num_entities[dim]; }

    /// Clear all data
    void clear();

    /// Initialize topology of given maximum dimension
    void init(uint dim);

    /// Set size for given dimension (number of entities)
    void init(uint dim, uint size);

    /// Set connectivity for given pair of dimensions
    void set(uint d0, uint d1, Array< Array<uint> >& connectivity);

    /// Display data
    void disp() const;

  private:

    // Topological dimension
    uint _dim;
  
    // Number of mesh entities for each topological dimension
    uint* num_entities;

    // Connections for pairs of topological dimensions
    MeshConnectivity** connectivity;
   
  };

}

#endif
