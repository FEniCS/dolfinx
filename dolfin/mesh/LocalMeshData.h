// Copyright (C) 2008 Ola Skavhaug
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
// Modified by Anders Logg, 2008-2009.
//
// First added:  2008-11-28
// Last changed: 2011-03-25
//
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2011.

#ifndef __LOCAL_MESH_DATA_H
#define __LOCAL_MESH_DATA_H

#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;
  class XMLLocalMeshDataDistributed;

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

  // FIXME: Provide a better public interface rather than using 'friend class'

  class LocalMeshData : public Variable
  {
  public:

    /// Create empty local mesh data
    LocalMeshData();

    /// Create local mesh data for given mesh
    LocalMeshData(const Mesh& mesh);

    /// Destructor
    ~LocalMeshData();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    /// Clear all data
    void clear();

    /// Copy data from mesh
    void extract_mesh_data(const Mesh& mesh);

    /// Broadcast mesh data from main process
    void broadcast_mesh_data();

    /// Receive mesh data from main process
    void receive_mesh_data();

    // Unpack received vertex coordinates
    void unpack_vertex_coordinates(const std::vector<double>& values);

    // Unpack received vertex indices
    void unpack_vertex_indices(const std::vector<uint>& values);

    // Unpack received cell vertices
    void unpack_cell_vertices(const std::vector<uint>& values);

    /// Coordinates for all vertices stored on local processor
    std::vector<std::vector<double> > vertex_coordinates;

    /// Global vertex indices for all vertices stored on local processor
    std::vector<uint> vertex_indices;

    /// Global vertex indices for all cells stored on local processor
    std::vector<std::vector<uint> > cell_vertices;

    /// Global cell numbers for all cells stored on local processor
    std::vector<uint> global_cell_indices;

    /// Global number of vertices
    uint num_global_vertices;

    /// Global number of cells
    uint num_global_cells;

    /// Number of vertices per cell
    uint num_vertices_per_cell;

    /// Geometrical dimension
    uint gdim;

    /// Topological dimension
    uint tdim;

    // A map from named mesh data to arrays
    std::map<std::string, std::vector<uint>* > arrays;

    // Friends
    friend class XMLLocalMeshData;
    friend class XMLLocalMeshDataDistributed;
    friend class MeshPartitioning;
    friend class GraphBuilder;
    friend class ParMETIS;
    friend class SCOTCH;

  };

}

#endif
