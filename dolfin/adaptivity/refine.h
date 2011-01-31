// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2011-01-31
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
  class GenericFunction;
  class Form;
  class VariationalProblem;

  //--- Refinement of meshes ---

  /// Refine mesh uniformly
  Mesh& refine(const Mesh& mesh);

  /// Refine mesh based on cell markers
  Mesh& refine(const Mesh& mesh,
               const MeshFunction<bool>& cell_markers);

  //--- Refinement of function spaces ---

  /// Refine function space uniformly
  FunctionSpace& refine(const FunctionSpace& space);

  /// Refine function space based on cell markers
  FunctionSpace& refine(const FunctionSpace& space,
                        const MeshFunction<bool>& cell_markers);

  /// Refine function space based on refined mesh
  FunctionSpace& refine(const FunctionSpace& space,
                       boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of functions ---

  /// Refine coefficient based on refined mesh
  Function& refine(const Function& function,
                   boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of forms ---

  /// Refine form based on refined mesh
  Form& refine(const Form& form,
               boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of variational problems ---

  /// Refine variational problem based on mesh
  VariationalProblem& refine(const VariationalProblem& problem,
                             boost::shared_ptr<const Mesh> refined_mesh);

}

#endif
