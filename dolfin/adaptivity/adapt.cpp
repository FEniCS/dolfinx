// Copyright (C) 2010-2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
// Modified by Marie E. Rognes, 2011.
//
// First added:  2010-02-10
// Last changed: 2011-03-12

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
#include <dolfin/plot/plot.h>
#include "ErrorControl.h"
#include "SpecialFacetFunction.h"
#include "adapt.h"

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
const dolfin::MeshFunction<dolfin::uint>& dolfin::adapt(const MeshFunction<uint>& mesh_function,
                                                boost::shared_ptr<const Mesh> refined_mesh)
{
  // FIXME: MeshFunction Hierarchical ok?
  // FIXME: Skip refinement if already refined
  // FIXME: Update according to shared_ptr changes in MeshFunction

  const Mesh& mesh = mesh_function.mesh();
  const uint dim = mesh.topology().dim();
  MeshFunction<uint> refined_mf(*refined_mesh, mesh_function.dim());

  // Extract parent encoding from refined mesh
  MeshFunction<uint>* parent = 0;
  if (mesh_function.dim() == dim)
  {
    parent = refined_mesh->data().mesh_function("parent_cell");
  } else if (mesh_function.dim() == (dim - 1))
  {
    parent = refined_mesh->data().mesh_function("parent_facet");
  } else
    dolfin_not_implemented();

  // Check that parent info exists
  if (!parent)
    error("Unable to extract information about parent mesh entites");

  // Map values of mesh function into refined mesh function
  for (uint i = 0; i < refined_mf.size(); i++)
    refined_mf[i] = mesh_function[(*parent)[i]];

  plot(refined_mf);

  // Return new mesh function
  return mesh_function; // FIXME

}
//-----------------------------------------------------------------------------
const dolfin::Mesh& dolfin::adapt(const Mesh& mesh)
{
  // Skip refinement if already refined
  if (mesh.has_child())
  {
    dolfin_debug("Mesh has already been refined, returning child mesh.");
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
const dolfin::Mesh& dolfin::adapt(const Mesh& mesh,
                                  const MeshFunction<bool>& cell_markers)
{
  // Skip refinement if already refined
  if (mesh.has_child())
  {
    dolfin_debug("Mesh has already been refined, returning child mesh.");
    return mesh.child();
  }

  // Call refinement algorithm
  boost::shared_ptr<Mesh> refined_mesh(new Mesh());
  LocalMeshRefinement::refine(*refined_mesh, mesh, cell_markers);

  // Set parent / child
  set_parent_child(mesh, refined_mesh);

  return *refined_mesh;
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space)
{
  // Refine mesh
  adapt(space.mesh());

  // Refine space
  adapt(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space,
                                           const MeshFunction<bool>& cell_markers)
{
  // Refine mesh
  adapt(space.mesh(), cell_markers);

  // Refine space
  adapt(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space,
                                           boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (space.has_child())
  {
    dolfin_debug("Function space has already been refined, returning child space.");
    return space.child();
  }

  // Create DOLFIN finite element and dofmap
  boost::shared_ptr<const FiniteElement> refined_element(space.element().create());
  boost::shared_ptr<const GenericDofMap> refined_dofmap(space.dofmap().copy(*refined_mesh));

  // Create new function space
  boost::shared_ptr<FunctionSpace> refined_space(new FunctionSpace(refined_mesh,
                                                                   refined_element,
                                                                   refined_dofmap));

  // Set parent / child
  set_parent_child(space, refined_space);

  return *refined_space;
}
//-----------------------------------------------------------------------------
const dolfin::Function& dolfin::adapt(const Function& function,
                                      boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (function.has_child())
  {
    dolfin_debug("Function has already been refined, returning child function.");
    return function.child();
  }

  // Refine function space
  boost::shared_ptr<const FunctionSpace> space = function.function_space_ptr();
  adapt(*space, refined_mesh);
  boost::shared_ptr<const FunctionSpace> refined_space = space->child_shared_ptr();

  // Create new function on refined space and interpolate
  boost::shared_ptr<Function> refined_function(new Function(refined_space));
  refined_function->interpolate(function);

  // Set parent / child
  set_parent_child(function, refined_function);

  return *refined_function;
}
//-----------------------------------------------------------------------------
const dolfin::Form& dolfin::adapt(const Form& form,
                                  boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (form.has_child())
  {
    dolfin_debug("Form has already been refined, returning child form.");
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
    adapt(space, refined_mesh);
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
      adapt(*function, refined_mesh);
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
const dolfin::VariationalProblem& dolfin::adapt(const VariationalProblem& problem,
                                                boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (problem.has_child())
  {
    dolfin_debug("Variational problem has already been refined, returning child problem.");
    return problem.child();
  }

  // Get data
  boost::shared_ptr<const Form> form_0 = problem.form_0_shared_ptr();
  boost::shared_ptr<const Form> form_1 = problem.form_1_shared_ptr();
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs = problem.bcs_shared_ptr();

  // Refine forms
  adapt(*form_0, refined_mesh);
  adapt(*form_1, refined_mesh);

  // Refine bcs
  std::vector<boost::shared_ptr<const BoundaryCondition> > refined_bcs;
  for (uint i = 0; i < bcs.size(); i++)
  {
    const DirichletBC* bc = dynamic_cast<const DirichletBC*>(bcs[i].get());
    if (bc != 0)
    {
      adapt(*bc, refined_mesh);
      refined_bcs.push_back(bc->child_shared_ptr());
    }
    else
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
const dolfin::DirichletBC& dolfin::adapt(const DirichletBC& bc,
                                         boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (bc.has_child())
  {
    dolfin_debug("DirichletBC has already been refined, returning child problem.");
    return bc.child();
  }

  // Refine function space
  boost::shared_ptr<const FunctionSpace> V = bc.function_space_ptr();
  adapt(*V, refined_mesh);

  // Extract but keep sub-domain
  boost::shared_ptr<const SubDomain> domain = bc.user_sub_domain_ptr();

  // Refine value
  const Function* g = dynamic_cast<const Function*>(bc.value_ptr().get());

  // Create refined boundary condition
  boost::shared_ptr<DirichletBC> refined_bc;
  if (g != 0)
  {
    adapt(*g, refined_mesh);
    refined_bc.reset(new DirichletBC(V->child_shared_ptr(),
                                     g->child_shared_ptr(),
                                     domain));
  }
  else
  {
    refined_bc.reset(new DirichletBC(V->child_shared_ptr(),
                                     bc.value_ptr(),
                                     domain));
  }
  // Set parent / child
  set_parent_child(bc, refined_bc);

  return *refined_bc;
}
//-----------------------------------------------------------------------------
const dolfin::ErrorControl& dolfin::adapt(const ErrorControl& ec,
                                          boost::shared_ptr<const Mesh> refined_mesh)
{
  // Skip refinement if already refined
  if (ec.has_child())
  {
    dolfin_debug("ErrorControl has already been refined, returning child problem.");
    return ec.child();
  }

  // Refine data
  adapt(*ec._a_star, refined_mesh);
  adapt(*ec._L_star, refined_mesh);
  adapt(*ec._residual, refined_mesh);
  adapt(*ec._a_R_T, refined_mesh);
  adapt(*ec._L_R_T, refined_mesh);
  adapt(*ec._a_R_dT, refined_mesh);
  adapt(*ec._L_R_dT, refined_mesh);
  adapt(*ec._eta_T, refined_mesh);

  // Create refined error control
  boost::shared_ptr<ErrorControl>
    refined_ec(new ErrorControl(ec._a_star->child_shared_ptr(),
                                ec._L_star->child_shared_ptr(),
                                ec._residual->child_shared_ptr(),
                                ec._a_R_T->child_shared_ptr(),
                                ec._L_R_T->child_shared_ptr(),
                                ec._a_R_dT->child_shared_ptr(),
                                ec._L_R_dT->child_shared_ptr(),
                                ec._eta_T->child_shared_ptr(),
                                ec._is_linear));

  // Set parent / child
  set_parent_child(ec, refined_ec);

  return *refined_ec;
}
//-----------------------------------------------------------------------------
