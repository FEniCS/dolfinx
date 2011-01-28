// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2011-01-29
//
// This file defines free functions for refinement.

#ifndef __DOLFIN_REFINE_H
#define __DOLFIN_REFINE_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;
  class FunctionSpace;
  class Function;
  class Form;

  //--- Refinement of meshes ---

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

  //--- Refinement of function spaces ---

  /// Create uniformly refined function space
  FunctionSpace refine(const FunctionSpace& space);

  /// Create locally refined function space
  FunctionSpace refine(const FunctionSpace& space,
                       const MeshFunction<bool>& cell_markers);

  /// Create refined function space for refined mesh
  FunctionSpace refine(const FunctionSpace& space,
                       const Mesh& refined_mesh);

  //--- Refinement of functions ---

  /// Create refined function for refined function space (interpolated)
  Function refine(const Function& function,
                  const FunctionSpace& refined_space);

  //--- Refinement of forms ---

  /// Create refined function for refined mesh
  Form refine(const Form& form,
              const Mesh& refined_mesh);

}

#endif
