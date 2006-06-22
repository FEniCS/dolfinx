// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-22

#ifndef __CELL_TYPE_H
#define __CELL_TYPE_H

#include <string>
#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class NewCell;
  class MeshEditor;

  /// This class provides a common interface for different cell types.
  /// Each cell type implements mesh functionality that is specific to
  /// a certain type of cell.

  class CellType
  {
  public:

    /// Enum for different cell types
    enum Type { interval, triangle, tetrahedron };

    /// Constructor
    CellType();

    /// Destructor
    virtual ~CellType();

    /// Create cell type from type (factory function)
    static CellType* create(Type type);

    /// Return type corresponding to given string
    static Type type(std::string type);

    /// Return type of cell
    virtual Type cellType() const = 0;

    /// Return type of cell for facets
    virtual Type facetType() const = 0;

    /// Return number of entitites of given topological dimension
    virtual uint numEntities(uint dim) const = 0;

    /// Return number of vertices for entity of given topological dimension
    virtual uint numVertices(uint dim) const = 0;

    /// Create entities e of given topological dimension from vertices v
    virtual void createEntities(uint** e, uint dim, const uint v[]) const = 0;

    /// Refine cell uniformly
    virtual void refineCell(NewCell& cell, MeshEditor& editor, uint& current_cell) const = 0;

    /// Return description of cell type
    virtual std::string description() const = 0;
    
  };

}

#endif
