// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#ifndef __NEW_TETRAHEDRON_H
#define __NEW_TETRAHEDRON_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for tetrahedrons.

  class NewTetrahedron : public CellType
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
