// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#ifndef __CELL_TYPE_H
#define __CELL_TYPE_H

#include <string>
#include <dolfin/constants.h>

namespace dolfin
{

  /// This class provides a common interface for different cell types.
  /// Each cell type implements mesh functionality that is specific to
  /// a certain type of cell.

  class CellType
  {
  public:

    /// Constructor
    CellType();

    /// Destructor
    virtual ~CellType();

    /// Create cell type from string (factory function)
    static CellType* create(std::string type);
    
    /// Return number of entitites of given topological dimension
    virtual uint size(uint dim) const = 0;

    /// Return description of cell type
    virtual std::string description() const = 0;
    
  };

}

#endif
