// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-12

#ifndef __NEW_TETRAHEDRON_H
#define __NEW_TETRAHEDRON_H

#include <dolfin/CellType.h>

namespace dolfin
{

  /// This class implements functionality for tetrahedral meshes.

  class NewTetrahedron : public CellType
  {
  public:

    /// Return type of cell
    inline Type type() const { return tetrahedron; }

    /// Return number of entitites of given topological dimension
    uint numEntities(uint dim) const;

    /// Return number of vertices for entity of given topological dimension
    uint numVertices(uint dim) const;

    /// Create entities of given topological dimension
    void createEntities(uint** entities, uint dim, const uint vertices[]);

    /// Refine mesh uniformly
    void refineUniformly(NewMesh& mesh);

    /// Return description of cell type
    std::string description() const;

  private:

  };

}

#endif
