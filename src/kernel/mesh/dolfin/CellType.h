// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-12

#ifndef __CELL_TYPE_H
#define __CELL_TYPE_H

#include <string>
#include <dolfin/constants.h>

namespace dolfin
{

  class NewMesh;

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

    // FIXME: Remove and add function for converting from string to type

    /// Create cell type from string (factory function)
    static CellType* create(std::string type);

    /// Create cell type from type (factory function)
    static CellType* create(Type type);

    /// Return type of cell
    virtual Type type() const = 0;

    /// Return number of entitites of given topological dimension
    virtual uint numEntities(uint dim) const = 0;

    /// Return number of vertices for entity of given topological dimension
    virtual uint numVertices(uint dim) const = 0;

    /// Create entities of given topological dimension
    virtual void createEntities(uint** entities, uint dim, const uint vertices[]) = 0;

    /// Refine mesh uniformly
    virtual void refineUniformly(NewMesh& mesh) = 0;

    /// Return description of cell type
    virtual std::string description() const = 0;
    
  };

}

#endif
