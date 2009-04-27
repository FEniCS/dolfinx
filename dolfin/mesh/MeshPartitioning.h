// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-01
// Last changed: 2009-04-16

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <iterator>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  template<typename InputIterator> void print_container(std::ostream& ostr, InputIterator itbegin, InputIterator itend, const std::string& delimiter)
  {
    std::copy(itbegin, itend, std::ostream_iterator<typename InputIterator::value_type>(ostr, delimiter.c_str()));
  }


  class LocalMeshData;

  /// This class partitions and distributes a mesh based on
  /// partitioned local mesh data. Note that the local mesh data will
  /// also be repartitioned and redistributed during the computation
  /// of the mesh partitioning.
  ///
  /// After partitioning, each process has a local mesh and set of
  /// mesh data that couples the meshes together.
  ///
  /// The following mesh data is created:
  ///
  /// 1. "global entity indices 0" (MeshFunction<uint>)
  ///
  /// This maps each local vertex to its global index.
  ///
  /// 2. "overlap" (std::map<uint, std::vector<uint> >)
  ///
  /// This maps each shared vertex to a list of the processes sharing
  /// the vertex.
  ///
  /// After partitioning, the function number_entities() may be called
  /// to create global indices for all entities of a given topological
  /// dimension. These are stored as mesh data (MeshFunction<uint>)
  /// named
  ///
  ///    "global entity indices 1"
  ///    "global entity indices 2"
  ///    etc

  class MeshPartitioning
  {
  public:

    /// Create a partitioned mesh based on partitioned local mesh data
    static void partition(Mesh& mesh, LocalMeshData& data);

    /// Create global entity indices for entities of dimension d
    static void number_entities(Mesh& mesh, uint d);

  private:

    // Compute cell partition
    static void compute_partition(std::vector<uint>& cell_partition,
                                  const LocalMeshData& data);

    // Distribute cells
    static void distribute_cells(LocalMeshData& data,
                                 const std::vector<uint>& cell_partition);

    // Distribute vertices
    static void distribute_vertices(LocalMeshData& data,
                                    std::map<uint, uint>& glob2loc);

    // Build mesh
    static void build_mesh(Mesh& mesh, const LocalMeshData& data,
                           std::map<uint, uint>& glob2loc);

    // Check if all entity vertices are in overlap
    static bool in_overlap(const std::vector<uint>& entity_vertices,
                           std::map<uint, std::vector<uint> >& overlap);

  };

}

#endif
