// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Ola Skavhaug, 2009.
// Modified by Garth N. Wells, 2010.
//
// First added:  2007-03-01
// Last changed: 2011-11-14

#ifndef __UFC_CELL_H
#define __UFC_CELL_H

#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/ParallelData.h>
#include <dolfin/fem/ufcexp.h>

namespace dolfin
{

  /// This class is simple wrapper for a UFC cell and provides
  /// a layer between a DOLFIN cell and a UFC cell. When run in
  /// parallel, it attempts to use global numbering.

  class UFCCell : public ufcexp::cell
  {
  public:

    /// Create UFC cell from DOLFIN cell
    UFCCell(const Cell& cell, bool use_global_indices=true) : ufcexp::cell(),
        use_global_indices(use_global_indices),
        num_vertices(0), num_higher_order_vertices(0)
    {
      init(cell);
    }

    /// Create UFC cell for first DOLFIN cell in mesh
    UFCCell(const Mesh& mesh, bool use_global_indices=true) : ufcexp::cell(),
        use_global_indices(use_global_indices),
        num_vertices(0), num_higher_order_vertices(0)
    {
      CellIterator cell(mesh);
      init(*cell);
    }

    /// Destructor
    ~UFCCell()
    { clear(); }

    /// Initialize UFC cell data
    void init(const Cell& cell)
    {
      // Clear old data
      clear();

      // Set cell shape
      switch (cell.type())
      {
      case CellType::interval:
        cell_shape = ufc::interval;
        num_vertices = 2;
        break;
      case CellType::triangle:
        cell_shape = ufc::triangle;
        num_vertices = 3;
        break;
      case CellType::tetrahedron:
        cell_shape = ufc::tetrahedron;
        num_vertices = 4;
        break;
      default:
        dolfin_error("UFCCell.h",
                     "create UFC cell wrapper",
                     "Unknown cell type (%d)", cell.type());
      }

      // Mesh
      const Mesh& mesh = cell.mesh();

      // Set topological dimension
      topological_dimension = mesh.topology().dim();

      // Set geometric dimension
      geometric_dimension = mesh.geometry().dim();

      // Allocate arrays for local entity indices
      entity_indices = new uint*[topological_dimension + 1];
      for (uint d = 0; d < topological_dimension; d++)
      {
        // Store number of cell entities allocated for (this can change between
        // init() and update() which is why it's stored)
        num_cell_entities.push_back(cell.num_entities(d));
        if (cell.num_entities(d) > 0)
          entity_indices[d] = new uint[cell.num_entities(d)];
        else
          entity_indices[d] = 0;
      }
      entity_indices[topological_dimension] = new uint[1];

      // Get global entity indices (if any)
      global_entities.resize(topological_dimension + 1);
      const ParallelData& parallel_data = mesh.parallel_data();
      for (uint d = 0; d <= topological_dimension; ++d)
      {
        if (parallel_data.have_global_entity_indices(d))
          global_entities[d] = &(parallel_data.global_entity_indices(d));
        else
          global_entities[d] = 0;
      }

      // Allocate vertex coordinates
      coordinates = new double*[num_vertices];

      // Allocate higher order vertex coordinates
      num_higher_order_vertices = cell.mesh().geometry().num_higher_order_vertices_per_cell();
      higher_order_coordinates = new double*[num_higher_order_vertices];

      // Update cell data
      update(cell);
    }

    // Clear UFC cell data
    void clear()
    {
      if (entity_indices)
      {
        for (uint d = 0; d <= topological_dimension; d++)
          delete [] entity_indices[d];
      }
      delete [] entity_indices;
      entity_indices = 0;

      global_entities.clear();

      delete [] coordinates;
      coordinates = 0;

      delete [] higher_order_coordinates;
      higher_order_coordinates = 0;

      cell_shape = ufc::interval;
      topological_dimension = 0;
      geometric_dimension = 0;
    }

    // Update cell entities and coordinates
    // Note: We use MeshEntity& rather than Cell& to avoid a gcc 4.4.1 warning
    void update(const MeshEntity& cell, int local_facet=-1)
    {
      dolfin_assert(cell.dim() == topological_dimension);

      // Note handling of local and global mesh entity indices.
      // If mappings from local to global entities are available in
      // MeshData ("global entity indices %d") then those are used.
      // Otherwise, local entities are used. It is the responsibility
      // of the DofMap class to create the local-to-global mapping of
      // entity indices when running in parallel. In that sense, this
      // class is not parallel aware. It just uses the local-to-global
      // mapping when it is available.

      // Set mesh identifier
      mesh_identifier = cell.mesh_id();

      // Set local facet (-1 means no local facet set)
      this->local_facet = local_facet;

      // Copy local entity indices from mesh
      const uint D = topological_dimension;
      for (uint d = 0; d < D; ++d)
      {
        for (uint i = 0; i < num_cell_entities[d]; ++i)
          entity_indices[d][i] = cell.entities(d)[i];
      }

      // Set cell index
      entity_indices[D][0] = cell.index();
      index = cell.index();

      // Map to global entity indices (if any)
      for (uint d = 0; d < D; ++d)
      {
        if (use_global_indices && global_entities[d])
        {
          for (uint i = 0; i < num_cell_entities[d]; ++i)
            entity_indices[d][i] = (*global_entities[d])[entity_indices[d][i]];
        }
      }
      if (use_global_indices && global_entities[D])
        entity_indices[D][0] = (*global_entities[D])[entity_indices[D][0]];

      // Set vertex coordinates
      const uint* vertices = cell.entities(0);
      for (uint i = 0; i < num_vertices; i++)
        coordinates[i] = const_cast<double*>(cell.mesh().geometry().x(vertices[i]));

      // Set higher order vertex coordinates
      if (num_higher_order_vertices > 0)
      {
        const uint current_cell_index = cell.index();
        const uint* higher_order_vertex_indices = cell.mesh().geometry().higher_order_cell(current_cell_index);
        for (uint i = 0; i < num_higher_order_vertices; i++)
          higher_order_coordinates[i] = const_cast<double*>(cell.mesh().geometry().higher_order_x(higher_order_vertex_indices[i]));
      }
    }

  private:

    // True it global entity indices should be used
    const bool use_global_indices;

    // Number of cell vertices
    uint num_vertices;

    // Number of higher order cell vertices
    uint num_higher_order_vertices;

    // Mappings from local to global entity indices (if any)
    std::vector<const MeshFunction<uint>* > global_entities;

    // Number of cell entities of dimension d at initialisation
    std::vector<uint> num_cell_entities;

  };

}

#endif
