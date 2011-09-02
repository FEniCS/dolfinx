// Copyright (C) 2010 Anders Logg, Marie Rognes and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-02-10
// Last changed: 2011-09-01
//
// This file defines free functions for refinement/adaption of meshes,
// function spaces, functions etc.

#ifndef __DOLFIN_ADAPT_H
#define __DOLFIN_ADAPT_H

#include <vector>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  template <class T> class MeshFunction;
  class FunctionSpace;
  class GenericFunction;
  class DirichletBC;
  class Form;
  class LinearVariationalProblem;
  class NonlinearVariationalProblem;
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

  /// Refine Function based on refined mesh
  const Function& adapt(const Function& function,
                        boost::shared_ptr<const Mesh> refined_mesh);

  /// Refine GenericFunction based on refined mesh
  const GenericFunction& adapt(const GenericFunction& function,
                               boost::shared_ptr<const Mesh> refined_mesh);

  /// Refine mesh function<uint> based on mesh
  const MeshFunction<dolfin::uint>& adapt(const MeshFunction<uint>& mesh_function,
                                  boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of boundary conditions ---

  /// Refine Dirichlet bc based on refined mesh
  const DirichletBC& adapt(const DirichletBC& bc,
                           boost::shared_ptr<const Mesh> refined_mesh,
                           const FunctionSpace& S);

  /// Helper function for refinement of boundary conditions
  void adapt_markers(std::vector<std::pair<uint, uint> >& refined_markers,
                     const Mesh& refined_mesh,
                     const std::vector<std::pair<uint, uint> >& markers,
                     const Mesh& mesh);

  //--- Refinement of forms ---

  /// Refine form based on refined mesh
  const Form& adapt(const Form& form,
                    boost::shared_ptr<const Mesh> refined_mesh);

  //--- Refinement of variational problems ---

  /// Refine linear variational problem based on mesh
  const LinearVariationalProblem& adapt(const LinearVariationalProblem& problem,
                                        boost::shared_ptr<const Mesh> refined_mesh);

  /// Refine nonlinear variational problem based on mesh
  const NonlinearVariationalProblem& adapt(const NonlinearVariationalProblem& problem,
                                           boost::shared_ptr<const Mesh> refined_mesh);

  /// Refine error control object based on mesh
  const ErrorControl& adapt(const ErrorControl& ec,
                            boost::shared_ptr<const Mesh> refined_mesh);


}

#endif
