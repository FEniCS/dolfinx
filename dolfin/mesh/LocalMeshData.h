// Copyright (C) 2008 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-17
//
// Modified by Anders Logg, 2008.

#ifndef __LOCAL_MESH_DATA_H
#define __LOCAL_MESH_DATA_H

#include <vector>
#include <dolfin/common/types.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;
  class XMLLocalMeshData;

  /// This class stores mesh data on a local processor corresponding
  /// to a portion of a (larger) global mesh.
  ///
  /// Note that the data stored in this class does typically not
  /// correspond to a topologically connected mesh; it merely stores a
  /// list of vertex coordinates, a list of cell-vertex mappings and a
  /// list of global vertex numbers for the locally stored vertices.
  ///
  /// It is typically used for parsing meshes in parallel from mesh
  /// XML files. After local mesh data has been parsed on each
  /// processor, a subsequent repartitioning takes place: first a
  /// geometric partitioning of the vertices followed by a
  /// redistribution of vertex and cell data, and then a topological
  /// partitioning again followed by redistribution of vertex and cell
  /// data, at that point corresponding to topologically connected
  /// meshes instead of local mesh data.

  class LocalMeshData
  {
  public:

    /// Create empty local mesh data
    LocalMeshData();

    /// Create local mesh data for given mesh
    LocalMeshData(const Mesh& mesh);

    /// Destructor
    ~LocalMeshData();

    /// Define XMLHandler for use in new XML reader/writer
    typedef XMLLocalMeshData XMLHandler;

  private:

    /// Clear all data
    void clear();

    /// Coordinates for all vertices stored on local processor
    std::vector<std::vector<double> > vertex_coordinates;

    /// Global vertex indices for all vertices stored on local processor
    std::vector<uint> vertex_indices;

    /// Global vertex indices for all cells stored on local processor
    std::vector<std::vector<uint> > cell_vertices;

    /// Global number of vertices
    uint num_global_vertices;

    /// Global number of cells
    uint num_global_cells;

    /// Number of processes
    uint num_processes;

    /// Local processes number
    uint process_number;

    /// Geometrical dimension
    uint gdim;

    /// Topological dimension
    uint tdim;

    /// Cell Type
    CellType* cell_type;

    // Friends
    friend class XMLLocalMeshData;
    friend class MeshPartitioning;

  };

}

#endif
