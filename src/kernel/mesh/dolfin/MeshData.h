// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-08
// Last changed: 2006-11-30

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <dolfin/MeshTopology.h>
#include <dolfin/MeshGeometry.h>
#include <dolfin/CellType.h>

namespace dolfin
{

  /// The class MeshData is a container for mesh data, including
  /// topology (mesh entities and connectivity) and geometry.
  ///
  /// For parallel meshes, each processor stores the local mesh
  /// data in a local MeshData object.
  
  class MeshData
  {
  public:
    
    /// Create empty mesh data
    MeshData();

    /// Copy constructor
    MeshData(const MeshData& data);
    
    /// Destructor
    ~MeshData();

    /// Assignment
    const MeshData& operator= (const MeshData& data);

    /// Clear all data
    void clear();

    /// Display data
    void disp() const;

    // Mesh topology
    MeshTopology topology;

    // Mesh geometry
    MeshGeometry geometry;

    // Cell type
    CellType* cell_type;
    
  private:
    friend class MPIMeshCommunicator;
  };

}

#endif
