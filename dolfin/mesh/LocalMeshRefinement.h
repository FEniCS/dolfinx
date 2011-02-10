// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-07

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

namespace dolfin
{

  class Mesh;
  template<class T> class MeshFunction;

  /// This class is provides functionality for local (adaptive) mesh
  /// refinement. It is a wrapper for various algorithms for local
  /// mesh refinement implemented as part of DOLFIN and it delegates
  /// the refinement to a particular refinement algorithm based on the
  /// value of the global parameter "refinement_algorithm".

  class LocalMeshRefinement
  {
  public:

    /// Refine mesh based on cell markers
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_markers);

  };

}

#endif
