// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
// Modified by Marie E. Rognes, 2011.
//
// First added:  2010-02-10
// Last changed: 2011-02-09

#include <boost/shared_ptr.hpp>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/LocalMeshRefinement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/UniformMeshRefinement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/VariationalProblem.h>
#include "ErrorControl.h"
#include "SpecialFacetFunction.h"
#include "refine.h"

using namespace dolfin;

// Common function for setting parent/child
template <class T>
void set_parent_child(const T& parent, boost::shared_ptr<T> child)
{
  // Use a const_cast so we can set the parent/child
  T& _parent = const_cast<T&>(parent);

  // Set parent/child
  _parent.set_child(child);
  child->set_parent(reference_to_no_delete_pointer(_parent));
}

//-----------------------------------------------------------------------------
const dolfin::Mesh& dolfin::refine(const Mesh& mesh)
{
  // Skip refinement if already refined
  if (mesh.has_child())
  {
    info("Mesh has already been refined, returning child mesh.");
    return mesh.child();
  }

  // Refine uniformly
  boost::shared_ptr<Mesh> refined_mesh(new Mesh());
  UniformMeshRefinement::refine(*refined_mesh, mesh);

  // Set parent / child
  set_parent_child(mesh, refined_mesh);

  return *refined_mesh;
}
//-----------------------------------------------------------------------------
const dolfin::Mesh& dolfin::refine(const Mesh& mesh,
                                   const MeshFunction<bool>& cell_markers)
{
  // Skip refinement if already refined
  if (mesh.has_child())
  {
    info("Mesh has already been refined, returning child mesh.");
    return mesh.child();
  }

  // Count the number of marked cells
  uint n0 = mesh.num_cells();
  uint n = 0;
  for (uint i = 0; i < cell_markers.size(); i++)
    if (cell_markers[i])
      n++;
  info("%d cells out of %d marked for refinement (%.1f%%).",
       n, n0, 100.0 * static_cast<double>(n) / static_cast<double>(n0));

  // Call refinement algorithm
  boost::shared_ptr<Mesh> refined_mesh(new Mesh());
  LocalMeshRefinement::refineRecursivelyByEdgeBisection(*refined_mesh,
                                                        mesh,
                                                        cell_markers);

  // Report the number of refined cells
  uint n1 = refined_mesh->num_cells();
  info("Number of cells increased from %d to %d (%.1f%% increase).",
       n0, n1, 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  // Set parent / child
  set_parent_child(mesh, refined_mesh);

  return *refined_mesh;
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space)
{
  // Refine mesh
  refine(space.mesh());

  // Refine space
  refine(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space,
                                            const MeshFunction<bool>& cell_markers)
{
  // Refine mesh
  refine(space.mesh(), cell_markers);

  // Refine space
  refine(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space,
                                            boost::shared_ptr<const Mesh> refined_mesh)
{
  cout << "Refining function space." << endl;

#ifndef UFC_DEV
  info("UFC_DEV compiler flag is not set.");
  error("Refinement of function spaces relies on the development version of UFC.");
  return const_cast<FunctionSpace&>(space);
#else

  // Skip refinement if already refined
  if (space.has_child())
  {
    info("Function space has already been refined, returning child space.");
    return space.child();
  }

  // Get DofMap (GenericDofMap does not know about ufc::dof_map)
  const DofMap* dofmap = dynamic_cast<const DofMap*>(&space.dofmap());
  if (!dofmap)
  {
    info("FunctionSpace is defined by a non-stand dofmap.");
    error("Unable to refine function space.");
  }

  // Create new copies of UFC finite element and dofmap
  boost::shared_ptr<ufc::finite_element> ufc_element(space.element().ufc_element()->create());
  boost::shared_ptr<ufc::dof_map> ufc_dofmap(dofmap->ufc_dofmap()->create());

  // Create DOLFIN finite element and dofmap
  boost::shared_ptr<const FiniteElement> refined_element(new FiniteElement(ufc_element));
  boost::shared_ptr<const DofMap> refined_dofmap(new DofMap(ufc_dofmap, *refined_mesh));

  // Create new function space
  boost::shared_ptr<FunctionSpace> refined_space(new FunctionSpace(refined_mesh,
                                                                   refined_element,
                                                                   refined_dofmap));

  // Set parent / child
  set_parent_child(space, refined_space);

  return *refined_space;

#endif
}
//-----------------------------------------------------------------------------
const dolfin::Function& dolfin::refine(const Function& function,
                                       boost::shared_ptr<const Mesh> refined_mesh)
{
  cout << "Refining function." << endl;

  // Skip refinement if already refined
  if (function.has_child())
  {
    info("Function has already been refined, returning child function.");
    return function.child();
  }

  // Refine function space
  boost::shared_ptr<const FunctionSpace> space = function.function_space_ptr();
  refine(*space, refined_mesh);
  boost::shared_ptr<const FunctionSpace> refined_space = space->child_shared_ptr();

  // Create new function on refined space and interpolate
  boost::shared_ptr<Function> refined_function(new Function(refined_space));
  refined_function->interpolate(function);

  // Set parent / child
  set_parent_child(function, refined_function);

  return *refined_function;
}
//-----------------------------------------------------------------------------
const dolfin::Form& dolfin::refine(const Form& form,
                                   boost::shared_ptr<const Mesh> refined_mesh)
{
  cout << "Refining form." << endl;

  // Skip refinement if already refined
  if (form.has_child())
  {
    info("Form has already been refined, returning child form.");
    return form.child();
  }

  // Get data
  std::vector<boost::shared_ptr<const FunctionSpace> > spaces = form.function_spaces();
  std::vector<boost::shared_ptr<const GenericFunction> > coefficients = form.coefficients();
  boost::shared_ptr<const ufc::form> ufc_form = form.ufc_form_shared_ptr();

  // Refine function spaces
  std::vector<boost::shared_ptr<const FunctionSpace> > refined_spaces;
  for (uint i = 0; i < spaces.size(); i++)
  {
    const FunctionSpace& space = *spaces[i];
    refine(space, refined_mesh);
    refined_spaces.push_back(space.child_shared_ptr());
  }


  // Refine coefficients
  std::vector<boost::shared_ptr<const GenericFunction> > refined_coefficients;
  for (uint i = 0; i < coefficients.size(); i++)
  {
    // Try casting to Function
    const Function* function = dynamic_cast<const Function*>(coefficients[i].get());

    // If we have a Function, refine
    if (function != 0)
    {
      refine(*function, refined_mesh);
      refined_coefficients.push_back(function->child_shared_ptr());
      continue;
    }

    // If not function, just reuse the same coefficient
    refined_coefficients.push_back(coefficients[i]);
  }

  /// Create new form (constructor used from Python interface)
  boost::shared_ptr<Form> refined_form(new Form(ufc_form,
                                                refined_spaces,
                                                refined_coefficients));

  /// Attach mesh
  refined_form->set_mesh(refined_mesh);

  // Set parent / child
  set_parent_child(form, refined_form);

  return *refined_form;
}
//-----------------------------------------------------------------------------
const dolfin::VariationalProblem& dolfin::refine(const VariationalProblem& problem,
                                                 boost::shared_ptr<const Mesh> refined_mesh)
{
  cout << "Refining variational problem." << endl;

  // Skip refinement if already refined
  if (problem.has_child())
  {
    info("Variational problem has already been refined, returning child problem.");
    return problem.child();
  }

  // Get data
  boost::shared_ptr<const Form> form_0 = problem.form_0_shared_ptr();
  boost::shared_ptr<const Form> form_1 = problem.form_1_shared_ptr();
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs = problem.bcs_shared_ptr();

  // Refine forms
  refine(*form_0, refined_mesh);
  refine(*form_1, refined_mesh);

  // Refine bcs
  std::vector<boost::shared_ptr<const BoundaryCondition> > refined_bcs;
  for (uint i = 0; i < bcs.size(); i++)
  {
    const DirichletBC* bc = dynamic_cast<const DirichletBC*>(bcs[i].get());
    if (bc != 0)
    {
      refine(*bc, refined_mesh);
      refined_bcs.push_back(bc->child_shared_ptr());
    } else
      error("Refinement of bcs only implemented for DirichletBCs!");
  }

  // FIXME: Skipping mesh functions

  // Create new problem
  boost::shared_ptr<VariationalProblem>
    refined_problem(new VariationalProblem(form_0->child_shared_ptr(),
                                           form_1->child_shared_ptr(),
                                           refined_bcs,
                                           0, 0, 0));

  // Set parent / child
  set_parent_child(problem, refined_problem);

  return *refined_problem;
}
//-----------------------------------------------------------------------------
const dolfin::DirichletBC& dolfin::refine(const DirichletBC& bc,
                                          boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (bc.has_child())
  {
    info("DirichletBC has already been refined, returning child problem.");
    return bc.child();
  }

  // Refine function space
  boost::shared_ptr<const FunctionSpace> V = bc.function_space_ptr();
  refine(*V, refined_mesh);

  // Refine value
  const Function& g(dynamic_cast<const Function&>(bc.value()));
  refine(g, refined_mesh);

  // Extract but keep sub-domain
  boost::shared_ptr<const SubDomain> domain = bc.user_sub_domain_ptr();

  // Create refined boundary condition
  boost::shared_ptr<DirichletBC>
    refined_bc(new DirichletBC(V->child_shared_ptr(), g.child_shared_ptr(),
                               domain));

  // Set parent / child
  set_parent_child(bc, refined_bc);

  return *refined_bc;
}
//-----------------------------------------------------------------------------
dolfin::ErrorControl& dolfin::refine(ErrorControl& ec,
                                     boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (ec.has_child())
  {
    info("ErrorControl has already been refined, returning child problem.");
    return ec.child();
  }

  // Refine data
  refine(*ec._a_star, refined_mesh);
  refine(*ec._L_star, refined_mesh);
  refine(*ec._residual, refined_mesh);
  refine(*ec._a_R_T, refined_mesh);
  refine(*ec._L_R_T, refined_mesh);
  refine(*ec._a_R_dT, refined_mesh);
  refine(*ec._L_R_dT, refined_mesh);
  // Don't need to refine *ec._eta_T

  // Create refined error control
  boost::shared_ptr<ErrorControl>
    refined_ec(new ErrorControl(ec._a_star->child_shared_ptr(),
                                ec._L_star->child_shared_ptr(),
                                ec._residual->child_shared_ptr(),
                                ec._a_R_T->child_shared_ptr(),
                                ec._L_R_T->child_shared_ptr(),
                                ec._a_R_dT->child_shared_ptr(),
                                ec._L_R_dT->child_shared_ptr(),
                                ec._eta_T,
                                ec._is_linear));

  // Set parent / child
  set_parent_child(ec, refined_ec);

  return *refined_ec;
}
//-----------------------------------------------------------------------------
