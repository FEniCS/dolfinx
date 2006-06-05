// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#ifndef __NEW_TRIANGLE_H
#define __NEW_TRIANGLE_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// Documentation of class

  class NewTriangle : public CellType
  {
  public:

    /// Return number of entitites of given topological dimension
    uint size(uint dim) const;

    /// Return description of cell type
    std::string description() const;

  private:

  };

}

#endif
