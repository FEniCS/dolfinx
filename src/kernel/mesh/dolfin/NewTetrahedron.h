// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-06

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
    uint numEntities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint numVertices(uint dim) const;

    /// Create entities of given topological dimension
    void createEntities(Array<Array<uint> >& entities, uint dim, const uint vertices[]);

    /// Return description of cell type
    std::string description() const;

  private:

  };

}

#endif
