// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#ifndef __INTERVAL_H
#define __INTERVAL_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for intervals.

  class Interval : public CellType
  {
  public:

    /// Return number of entitites of given topological dimension
    uint size(uint dim) const;

    /// Return description of cell type
    std::string description() const;

  };

}

#endif
