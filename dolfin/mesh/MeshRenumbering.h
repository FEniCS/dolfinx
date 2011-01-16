// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2010-11-27
// Last changed: 2011-01-16

#ifndef __MESH_RENUMBERING_H
#define __MESH_RENUMBERING_H

#include <boost/tuple/tuple.hpp>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Mesh;

  /// This class implements renumbering algorithms for meshes.

  class MeshRenumbering
  {
  public:

    /// Renumber mesh entities by coloring. This function is currently
    /// restricted to renumbering by cell coloring. The cells
    /// (cell-vertex connectivity) and the coordinates of the mesh are
    /// renumbered to improve the locality within each color. It is
    /// assumed that the mesh has already been colored and that only
    /// cell-vertex connectivity exists as part of the mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Mesh to be renumbered.
    static void renumber_by_color(Mesh& mesh,
                                  boost::tuple<uint, uint, uint> coloring);

  };

}

#endif
