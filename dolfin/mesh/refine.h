// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-02-10
// Last changed: 2010-02-26
//
// This file defines free functions for mesh refinement.
//

#ifndef __DOLFIN_REFINE_H
#define __DOLFIN_REFINE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;

  /// Create uniformly refined mesh
  Mesh refine(const Mesh& mesh);

  /// Create uniformly refined mesh
  void refine(Mesh& refined_mesh,
              const Mesh& mesh);

  /// Create locally refined mesh
  Mesh refine(const Mesh& mesh,
              const MeshFunction<bool>& cell_markers);

  /// Create locally refined mesh
  void refine(Mesh& refined_mesh,
              const Mesh& mesh,
              const MeshFunction<bool>& cell_markers);

}

#endif
