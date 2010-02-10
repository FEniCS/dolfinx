// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:
//
// This file defines free functions for mesh refinement.
//

#ifndef __REFINE_H
#define __REFINE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;

  /// Create uniformly refined mesh
  Mesh refine(const Mesh& mesh);

  /// Create locally refined mesh
  Mesh refine(const Mesh& mesh, const MeshFunction<bool>& cell_markers);

}

#endif
