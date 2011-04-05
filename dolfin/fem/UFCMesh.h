// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-01
// Last changed: 2007-03-01

#ifndef __UFC_MESH_H
#define __UFC_MESH_H

#include <ufc.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// This class is simple wrapper for a UFC mesh and provides
  /// a layer between a DOLFIN mesh and a UFC mesh.

  class UFCMesh : public ufc::mesh
  {
  public:

    /// Create empty UFC mesh
    UFCMesh() : ufc::mesh() {}

    /// Copy constructor
    UFCMesh(const UFCMesh& mesh) : ufc::mesh()
    {
      // Set topological dimension
      topological_dimension = mesh.topological_dimension;

      // Set geometric dimension
      geometric_dimension = mesh.geometric_dimension;

      num_entities = new uint[topological_dimension + 1];
      for (uint d = 0; d <= topological_dimension; d++)
        num_entities[d] = mesh.num_entities[d];
    }

    /// Create UFC mesh from DOLFIN mesh
    UFCMesh(const Mesh& mesh) : ufc::mesh()
    {
      init(mesh);
    }

    /// Destructor
    ~UFCMesh()
    {
      clear();
    }

    /// Initialize UFC cell data
    void init(const Mesh& mesh)
    {
      // Clear old data
      clear();

      // Set topological dimension
      topological_dimension = mesh.topology().dim();

      // Set geometric dimension
      geometric_dimension = mesh.geometry().dim();

      // Set number of entities for each topological dimension
      num_entities = new uint[mesh.topology().dim() + 1];

      // Use number of global entities if available (when running in parallel)
      const std::vector<uint>& num_global_entities = mesh.parallel_data().num_global_entities();
      if (num_global_entities.size() > 0)
        std::copy(num_global_entities.begin(), num_global_entities.end(), num_entities);
      else
      {
        for (uint d = 0; d <= mesh.topology().dim(); d++)
          num_entities[d] = mesh.size(d);
      }
    }

    // Clear UFC cell data
    void clear()
    {
      topological_dimension = 0;
      geometric_dimension = 0;

      delete [] num_entities;
      num_entities = 0;
    }

  };

}

#endif
