// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-06-05

#ifndef __NEW_MESH_DATA_H
#define __NEW_MESH_DATA_H

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
  
  class NewMeshData
  {
  public:
    
    /// Create empty mesh data
    NewMeshData();
    
    /// Destructor
    ~NewMeshData();

    /// Clear all data
    void clear();

    /// Display data
    void disp() const;

  private:

    // Friends
    friend class NewMesh;
    friend class MeshEditor;
    friend class MeshAlgorithms;
    friend class MeshEntityIterator;
    
    // Mesh topology
    MeshTopology topology;

    // Mesh geometry
    MeshGeometry geometry;

    // Cell type
    CellType* cell_type;
    
  };

}

#endif
