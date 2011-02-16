// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
// Modified by Marie E. Rognes, 2011.
//
// First added:  2010-02-10
// Last changed: 2011-02-16
//
// This file defines free functions for refinement/adaption of meshes,
// function spaces, functions etc.

#ifndef __DOLFIN_ADAPT_H
#define __DOLFIN_ADAPT_H

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;
  class FunctionSpace;
  class GenericFunction;
  class DirichletBC;
  class Form;
  class VariationalProblem;
  class ErrorControl;

  //--- Refinement of meshes ---

  /// Refine mesh uniformly
  const Mesh& adapt(const Mesh& mesh);

  /// Refine mesh based on cell markers
  const Mesh& adapt(const Mesh& mesh, const MeshFunction<bool>& cell_markers);

  //--- Refinement of function spaces ---

  /// Refine function space uniformly
  const FunctionSpace& adapt(const FunctionSpace& space);

  /// Refine function space based on cell markers
  const FunctionSpace& adapt(const FunctionSpace& space,
                       const MeshFunction<bool>& cell_markers);

  /// Refine function space based on refined mesh
  const FunctionSpace& adapt(const FunctionSpace& space,
                       boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of functions ---

  /// Refine coefficient based on refined mesh
  const Function& adapt(const Function& function,
                  boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of boundary conditions ---

  /// Refine Dirichlet bc based on refined mesh
  const DirichletBC& adapt(const DirichletBC& bc,
                     boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of forms ---

  /// Refine form based on refined mesh
  const Form& adapt(const Form& form,
              boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of variational problems ---

  /// Refine variational problem based on mesh
  const VariationalProblem& adapt(const VariationalProblem& problem,
                            boost::shared_ptr<const Mesh> refined_mesh);

  /// Refine error control object based on mesh
  const ErrorControl& adapt(const ErrorControl& ec,
                            boost::shared_ptr<const Mesh> refined_mesh);

}

#endif
