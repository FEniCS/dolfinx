// Copyright (C) 2007 Magnus Vikstrom
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Niclas Jansson, 2008.
//
// First added:  2007-04-24
// Last changed: 2008-12-02

#ifndef __MESH_PARTITIONING_H
#define __MESH_PARTITIONING_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class LocalMeshData;

  /// This class provides a set of functions to partition a Mesh

  class MeshPartition
  {
  public:

    /// Create a partitioned mesh based on partitioned local mesh data
    static void partition(Mesh& mesh, const LocalMeshData& data);

    /// Partition a mesh into num_part partitions
    static void partition(Mesh& mesh, MeshFunction<uint>& partitions, uint num_partitions);

    /// Partition mesh in meshfile into num_partitions meshes with subdomain and facet overlap data attached 
    /// Outfile names are given running numbering <meshfile>_i.tar.gz for i in [0, num_partitions).
    static void partition(std::string meshfile, uint num_partitions);


    /// Partition a mesh into COMM_SIZE partitions
    static void partition(Mesh& mesh, MeshFunction<uint>& partitions);
    
    /// Partition a mesh into COMM_SIZE partitions, using a weighted dual graph
    static void partition(Mesh& mesh, 
			  MeshFunction<uint>& partitions,
			  MeshFunction<uint>& weight);
    
  private:

    /// Common Metis interface
    static void partitionCommonMetis(Mesh& mesh,
				     MeshFunction<uint>& partitions,
				     MeshFunction<uint>* weight);

  };

}

#endif
