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
    UFCCell(const Cell& cell) : ufcexp::cell(), num_vertices(0)
    {
      init(cell);
    }

    /// Create UFC cell for first DOLFIN cell in mesh
    UFCCell(const Mesh& mesh) : ufcexp::cell(), num_vertices(0)
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

      // Set orientation (default to 0)
      orientation = 0;

      // Set topological dimension
      topological_dimension = mesh.topology().dim();

      // Set geometric dimension
      geometric_dimension = mesh.geometry().dim();

      // Allocate arrays for local entity indices
      entity_indices = new std::size_t*[topological_dimension + 1];
      for (std::size_t d = 0; d < topological_dimension; d++)
      {
        // Store number of cell entities allocated for (this can change
        // between init() and update() which is why it's stored)
        num_cell_entities.push_back(cell.num_entities(d));
        if (cell.num_entities(d) > 0)
          entity_indices[d] = new std::size_t[cell.num_entities(d)];
        else
          entity_indices[d] = 0;
      }
      entity_indices[topological_dimension] = new std::size_t[1];

      // Allocate vertex coordinates
      coordinates = new double*[num_vertices];

      // Update cell data
      update(cell);
    }

    // Clear UFC cell data
    void clear()
    {
      if (entity_indices)
      {
        for (std::size_t d = 0; d <= topological_dimension; d++)
          delete [] entity_indices[d];
      }
      delete [] entity_indices;
      entity_indices = 0;

      delete [] coordinates;
      coordinates = 0;

      cell_shape = ufc::interval;
      topological_dimension = 0;
      geometric_dimension = 0;
      orientation = 0;
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

      // Set orientation
      this->orientation = cell.mesh().cell_orientations()[cell.index()];
      const std::size_t D = topological_dimension;

      const MeshTopology& topology = cell.mesh().topology();
      for (std::size_t d = 0; d < D; ++d)
      {
        //if (use_global_indices && topology.have_global_indices(d))
        if (topology.have_global_indices(d))
        {
          const std::vector<std::size_t>& global_indices = topology.global_indices(d);
          for (std::size_t i = 0; i < num_cell_entities[d]; ++i)
            entity_indices[d][i] = global_indices[cell.entities(d)[i]];
        }
        else
        {
          for (std::size_t i = 0; i < num_cell_entities[d]; ++i)
            entity_indices[d][i] = cell.entities(d)[i];
        }
      }

      // Set cell index
      //if (use_global_indices && topology.have_global_indices(D))
      //  entity_indices[D][0] = cell.global_index();
      //else
      entity_indices[D][0] = cell.index();

      // FIXME: Using the local cell index is inconsistent with UFC, but
      //        necessary to make DOLFIN run
      // Local cell index
      index = cell.index();

      // Set vertex coordinates
      const std::size_t* vertices = cell.entities(0);
      for (std::size_t i = 0; i < num_vertices; i++)
        coordinates[i] = const_cast<double*>(cell.mesh().geometry().x(vertices[i]));
    }

  private:

    // True if global entity indices should be used
    //const bool use_global_indices;

    // Number of cell vertices
    std::size_t num_vertices;

    // Number of cell entities of dimension d at initialisation
    std::vector<std::size_t> num_cell_entities;

  };

}

#endif
