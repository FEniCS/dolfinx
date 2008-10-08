// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-01
// Last changed: 2008-10-06

#ifndef __UFC_CELL_H
#define __UFC_CELL_H

#include <ufc.h>

#include <dolfin/common/types.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Cell.h>

namespace dolfin
{

  /// This class is simple wrapper for a UFC cell and provides
  /// a layer between a DOLFIN cell and a UFC cell.

  class UFCCell : public ufc::cell
  {
  public:
    
    /// Create emtpy UFC cell
    UFCCell() : ufc::cell(), num_vertices(0) {}

    /// Create UFC cell from DOLFIN cell
    UFCCell(Cell& cell) : ufc::cell(), num_vertices(0)
    {
      init(cell);
    }

    /// Create UFC cell for first DOLFIN cell in mesh
    UFCCell(Mesh& mesh) : ufc::cell(), num_vertices(0)
    {
      CellIterator cell(mesh);
      init(*cell);
    }

    /// Destructor
    ~UFCCell()
    {
      clear();
    }
    
    /// Initialize UFC cell data
    void init(Cell& cell)
    {
      // Clear old data
      clear();

      // Set cell shape
      switch ( cell.type() )
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
        error("Unknown cell type.");
      }

      // Set topological dimension
      topological_dimension = cell.mesh().topology().dim();

      // Set geometric dimension
      geometric_dimension = cell.mesh().geometry().dim();

      // Set entity indices
      entity_indices = new uint*[topological_dimension + 1];
      entity_indices[topological_dimension] = new uint[1];
      for (uint d = 0; d < topological_dimension; d++)
        entity_indices[d] = cell.entities(d);
      entity_indices[topological_dimension][0] = cell.index();

      /// Set vertex coordinates
      uint* vertices = cell.entities(0);
      coordinates = new double*[num_vertices];
      for (uint i = 0; i < num_vertices; i++)
        coordinates[i] = cell.mesh().geometry().x(vertices[i]);
    }

    // Clear UFC cell data
    void clear()
    {
      if (entity_indices)
      {
        delete [] entity_indices[topological_dimension];
        delete [] entity_indices;
      }
      entity_indices = 0;

      if (coordinates)
        delete [] coordinates;
      coordinates = 0;

      cell_shape = ufc::interval;
      topological_dimension = 0;
      geometric_dimension = 0;
    }

    // Update cell entities and coordinates
    inline void update(Cell& cell)
    {
      // Set entity indices
      for (uint d = 0; d < topological_dimension; d++)
        entity_indices[d] = cell.entities(d);
      entity_indices[topological_dimension][0] = cell.index();

      /// Set vertex coordinates
      const uint* vertices = cell.entities(0);
      for (uint i = 0; i < num_vertices; i++)
        coordinates[i] = cell.mesh().geometry().x(vertices[i]);
    }

  private:
    
    // Number of cell vertices
    uint num_vertices;

  };

}

#endif
