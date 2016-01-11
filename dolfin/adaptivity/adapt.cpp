// Copyright (C) 2010-2011 Anders Logg, Marie Rognes and Garth N. Wells
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
// Last changed: 2012-11-29

#include <map>
#include <memory>

#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/SpecialFacetFunction.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/refinement/refine.h>
#include "ErrorControl.h"
#include "adapt.h"

using namespace dolfin;

// Common function for setting parent/child
template <typename T>
void set_parent_child(const T& parent, std::shared_ptr<T> child)
{
  // Use a const_cast so we can set the parent/child
  T& _parent = const_cast<T&>(parent);

  // Set parent/child
  _parent.set_child(child);
  child->set_parent(reference_to_no_delete_pointer(_parent));
}
//-----------------------------------------------------------------------------
const Mesh& dolfin::adapt(const Mesh& mesh)
{
  // Skip refinement if already refined
  if (mesh.has_child())
  {
    dolfin_debug("Mesh has already been refined, returning child mesh.");
    return mesh.child();
  }

  // Refine uniformly
  std::shared_ptr<Mesh> adapted_mesh(new Mesh());
  refine(*adapted_mesh, mesh);

  // Initialize the entities initialized in mesh in adapted_mesh
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
    if (mesh.num_entities(d) != 0)
      adapted_mesh->init(d);

  // Set parent / child
  set_parent_child(mesh, adapted_mesh);

  return *adapted_mesh;
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
  std::shared_ptr<Mesh> adapted_mesh(new Mesh());
  refine(*adapted_mesh, mesh, cell_markers);

  // Initialize the entities initialized in mesh in adapted_mesh
  for (std::size_t d = 0; d <= mesh.topology().dim(); ++d)
    if (mesh.num_entities(d) != 0)
      adapted_mesh->init(d);

  // Set parent / child
  set_parent_child(mesh, adapted_mesh);

  return *adapted_mesh;
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space)
{
  dolfin_assert(space.mesh());

  // Refine mesh
  adapt(*space.mesh());

  // Refine space
  adapt(space, space.mesh()->child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space,
                                           const MeshFunction<bool>& cell_markers)
{
  dolfin_assert(space.mesh());

  // Refine mesh
  adapt(*space.mesh(), cell_markers);

  // Refine space
  adapt(space, space.mesh()->child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
const dolfin::FunctionSpace& dolfin::adapt(const FunctionSpace& space,
                                           std::shared_ptr<const Mesh> adapted_mesh)
{

  // Skip refinement if already refined and child's mesh is the same
  // as requested
  if (space.has_child()
      && adapted_mesh.get() == space.child().mesh().get())
  {
    dolfin_debug("Function space has already been refined, returning child space.");
    return space.child();
  }

  // Create DOLFIN finite element and dofmap
  dolfin_assert(space.dofmap());
  dolfin_assert(space.element());
  std::shared_ptr<const FiniteElement>
    refined_element(space.element()->create());
  std::shared_ptr<const GenericDofMap>
      refined_dofmap(space.dofmap()->create(*adapted_mesh));

  // Create new function space
  std::shared_ptr<FunctionSpace>
    refined_space(new FunctionSpace(adapted_mesh, refined_element, refined_dofmap));

  // Set parent / child
  set_parent_child(space, refined_space);

  return *refined_space;
}
//-----------------------------------------------------------------------------
const dolfin::Function& dolfin::adapt(const Function& function,
                                      std::shared_ptr<const Mesh> adapted_mesh,
                                      bool interpolate)
{
  // Skip refinement if already refined and if child's mesh matches
  // requested mesh
  if (function.has_child()
      && adapted_mesh.get() == function.child().function_space()->mesh().get())
  {
    dolfin_debug("Function has already been refined, returning child function.");
    return function.child();
  }

  // Refine function space
  std::shared_ptr<const FunctionSpace> space = function.function_space();
  dolfin_assert(space);
  adapt(*space, adapted_mesh);
  std::shared_ptr<const FunctionSpace>
    refined_space = space->child_shared_ptr();
  dolfin_assert(refined_space);

  // Create new function on refined space and interpolate
  std::shared_ptr<Function> refined_function(new Function(refined_space));
  if (interpolate)
    refined_function->interpolate(function);

  // Set parent / child
  set_parent_child(function, refined_function);

  return *refined_function;
}
//-----------------------------------------------------------------------------
const dolfin::GenericFunction& dolfin::adapt(const GenericFunction& function,
                                             std::shared_ptr<const Mesh> adapted_mesh)
{
  // Try casting to a function
  const Function* f = dynamic_cast<const Function*>(&function);
  if (f)
    return adapt(*f, adapted_mesh);
  else
    return function;
}
//-----------------------------------------------------------------------------
const dolfin::Form& dolfin::adapt(const Form& form,
                                  std::shared_ptr<const Mesh> adapted_mesh,
                                  bool adapt_coefficients)
{
  // Skip refinement if already refined
  if (form.has_child())
  {
    dolfin_debug("Form has already been refined, returning child form.");
    return form.child();
  }

  // Get data
  std::vector<std::shared_ptr<const FunctionSpace>>
    spaces = form.function_spaces();
  std::vector<std::shared_ptr<const GenericFunction>>
    coefficients = form.coefficients();
  std::shared_ptr<const ufc::form> ufc_form = form.ufc_form();

  // Refine function spaces
  std::vector<std::shared_ptr<const FunctionSpace>> refined_spaces;
  for (std::size_t i = 0; i < spaces.size(); i++)
  {
    const FunctionSpace& space = *spaces[i];
    adapt(space, adapted_mesh);
    refined_spaces.push_back(space.child_shared_ptr());
  }

  // Refine coefficients:
  std::vector<std::shared_ptr<const GenericFunction>> refined_coefficients;
  for (std::size_t i = 0; i < coefficients.size(); i++)
  {
    // Try casting to Function
    const Function* function
      = dynamic_cast<const Function*>(coefficients[i].get());

    if (function)
    {
      adapt(*function, adapted_mesh, adapt_coefficients);
      refined_coefficients.push_back(function->child_shared_ptr());
    }
    else
      refined_coefficients.push_back(coefficients[i]);
  }

  // Create new form (constructor used from Python interface)
  std::shared_ptr<Form> refined_form(new Form(ufc_form,
                                              refined_spaces,
                                              refined_coefficients));

  // Attach mesh
  refined_form->set_mesh(adapted_mesh);

  // Attached refined sub domains
  const MeshFunction<std::size_t>* cell_domains = form.cell_domains().get();
  if (cell_domains)
  {
    adapt(*cell_domains, adapted_mesh);
    refined_form->dx = cell_domains->child_shared_ptr();
  }
  const MeshFunction<std::size_t>* exterior_domains
    = form.exterior_facet_domains().get();
  if (exterior_domains)
  {
    adapt(*exterior_domains, adapted_mesh);
    refined_form->ds = exterior_domains->child_shared_ptr();
  }
  const MeshFunction<std::size_t>* interior_domains
    = form.interior_facet_domains().get();
  if (interior_domains)
  {
    adapt(*interior_domains, adapted_mesh);
    refined_form->dS = interior_domains->child_shared_ptr();
  }

  // Set parent / child
  set_parent_child(form, refined_form);

  return *refined_form;
}
//-----------------------------------------------------------------------------
const dolfin::LinearVariationalProblem&
dolfin::adapt(const LinearVariationalProblem& problem,
              std::shared_ptr<const Mesh> adapted_mesh)
{
  // Skip refinement if already refined
  if (problem.has_child())
  {
    dolfin_debug("Linear variational problem has already been refined, returning child problem.");
    return problem.child();
  }

  // Get data
  std::shared_ptr<const Form> a = problem.bilinear_form();
  std::shared_ptr<const Form> L = problem.linear_form();
  std::shared_ptr<const Function> u = problem.solution();
  std::vector<std::shared_ptr<const DirichletBC>> bcs = problem.bcs();

  // Refine forms
  dolfin_assert(a);
  dolfin_assert(L);
  adapt(*a, adapted_mesh);
  adapt(*L, adapted_mesh);

  // FIXME: Note const-cast here, don't know how to get around it

  // Refine solution variable
  dolfin_assert(u);
  adapt(*u, adapted_mesh);
  std::shared_ptr<Function> refined_u
    = reference_to_no_delete_pointer(const_cast<Function&>(u->child()));

  // Refine bcs
  std::shared_ptr<const FunctionSpace> V(problem.trial_space());
  std::vector<std::shared_ptr<const DirichletBC>> refined_bcs;
  for (std::size_t i = 0; i < bcs.size(); i++)
  {
    if (bcs[i] != 0)
    {
      dolfin_assert(V);
      adapt(*bcs[i], adapted_mesh, *V);
      refined_bcs.push_back(bcs[i]->child_shared_ptr());
    }
    else
    {
      dolfin_error("adapt.cpp",
                   "adapt linear variational problem",
                   "Only implemented for Dirichlet boundary conditions");
    }
  }

  // Create new problem
  dolfin_assert(a);
  dolfin_assert(L);
  dolfin_assert(u);
  std::shared_ptr<LinearVariationalProblem>
    refined_problem(new LinearVariationalProblem(a->child_shared_ptr(),
                                                 L->child_shared_ptr(),
                                                 refined_u,
                                                 refined_bcs));

  // Set parent / child
  set_parent_child(problem, refined_problem);

  return *refined_problem;
}
//-----------------------------------------------------------------------------
const dolfin::NonlinearVariationalProblem&
dolfin::adapt(const NonlinearVariationalProblem& problem,
              std::shared_ptr<const Mesh> adapted_mesh)
{
  // Skip refinement if already refined
  if (problem.has_child())
  {
    dolfin_debug("Nonlinear variational problem has already been refined, returning child problem.");
    return problem.child();
  }

  // Get data
  std::shared_ptr<const Form> F = problem.residual_form();
  std::shared_ptr<const Form> J = problem.jacobian_form();
  std::shared_ptr<const Function> u = problem.solution();
  std::vector<std::shared_ptr<const DirichletBC>> bcs = problem.bcs();

  // Refine forms
  dolfin_assert(F);
  adapt(*F, adapted_mesh);
  if (J)
    adapt(*J, adapted_mesh);

  // FIXME: Note const-cast here, don't know how to get around it

  // Refine solution variable
  dolfin_assert(u);
  adapt(*u, adapted_mesh);
  std::shared_ptr<Function> refined_u =
    reference_to_no_delete_pointer(const_cast<Function&>(u->child()));

  // Refine bcs
  std::shared_ptr<const FunctionSpace> V(problem.trial_space());
  std::vector<std::shared_ptr<const DirichletBC>> refined_bcs;
  for (std::size_t i = 0; i < bcs.size(); i++)
  {
    dolfin_assert(bcs[i] != 0);
    dolfin_assert(V);
    adapt(*bcs[i], adapted_mesh, *V);
    refined_bcs.push_back(bcs[i]->child_shared_ptr());
  }

  // Create new problem
  dolfin_assert(F);
  dolfin_assert(u);
  std::shared_ptr<NonlinearVariationalProblem> refined_problem;
  if (J)
  {
    refined_problem.reset(new NonlinearVariationalProblem(F->child_shared_ptr(),
                                                          refined_u,
                                                          refined_bcs,
                                                          J->child_shared_ptr()));
  }
  else
  {
    refined_problem.reset(new NonlinearVariationalProblem(F->child_shared_ptr(),
                                                          refined_u,
                                                          refined_bcs));
  }

  // Set parent / child
  set_parent_child(problem, refined_problem);

  return *refined_problem;
}
//-----------------------------------------------------------------------------
const dolfin::DirichletBC& dolfin::adapt(const DirichletBC& bc,
                                    std::shared_ptr<const Mesh> adapted_mesh,
                                    const FunctionSpace& S)
{
  dolfin_assert(adapted_mesh);

  // Skip refinement if already refined and child's mesh is the same
  // as requested
  if (bc.has_child()
      && adapted_mesh.get() == bc.child().function_space()->mesh().get())
  {
    dolfin_debug("DirichletBC has already been refined, returning child.");
    return bc.child();
  }

  std::shared_ptr<const FunctionSpace> W = bc.function_space();
  dolfin_assert(W);

  // Refine function space
  const std::vector<std::size_t> component = W->component();
  std::shared_ptr<const FunctionSpace> V;
  if (component.empty())
  {
    adapt(*W, adapted_mesh);
    V = W->child_shared_ptr();
  }
  else
  {
    adapt(S, adapted_mesh);
    V = S.child().sub(component);
  }

  // Get refined value
  const GenericFunction& g = adapt(*(bc.value()), adapted_mesh);
  std::shared_ptr<const GenericFunction>
    g_ptr(reference_to_no_delete_pointer(g));

  // Extract user_sub_domain
  std::shared_ptr<const SubDomain> user_sub_domain = bc.user_sub_domain();

  // Create refined boundary condition
  std::shared_ptr<DirichletBC> refined_bc;
  if (user_sub_domain)
  {
    // Use user defined sub domain if defined
    refined_bc.reset(new DirichletBC(V, g_ptr, user_sub_domain, bc.method()));
  }
  else
  {
    // Extract markers
    const std::vector<std::size_t>& markers = bc.markers();

    // Create refined markers
    dolfin_assert(W->mesh());
    std::vector<std::size_t> refined_markers;
    adapt_markers(refined_markers, *adapted_mesh, markers, *W->mesh());

    refined_bc.reset(new DirichletBC(V, g_ptr, refined_markers, bc.method()));
  }

  // Set parent / child
  set_parent_child(bc, refined_bc);

  return *refined_bc;
}
//-----------------------------------------------------------------------------
const dolfin::ErrorControl&
dolfin::adapt(const ErrorControl& ec,
              std::shared_ptr<const Mesh> adapted_mesh,
              bool adapt_coefficients)
{
  dolfin_assert(adapted_mesh);

  // Skip refinement if already refined
  if (ec.has_child())
  {
    dolfin_debug("ErrorControl has already been refined, returning child problem.");
    return ec.child();
  }

  // Refine data
  adapt(*ec._residual, adapted_mesh, adapt_coefficients);
  adapt(*ec._L_star, adapted_mesh, adapt_coefficients);
  adapt(*ec._a_star, adapted_mesh, adapt_coefficients);
  adapt(*ec._a_R_T, adapted_mesh, adapt_coefficients);
  adapt(*ec._L_R_T, adapted_mesh, adapt_coefficients);
  adapt(*ec._a_R_dT, adapted_mesh, adapt_coefficients);
  adapt(*ec._L_R_dT, adapted_mesh, adapt_coefficients);
  adapt(*ec._eta_T, adapted_mesh, adapt_coefficients);

  // Create refined error control
  std::shared_ptr<ErrorControl>
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
const dolfin::MeshFunction<std::size_t>&
  dolfin::adapt(const MeshFunction<std::size_t>& mesh_function,
                std::shared_ptr<const Mesh> adapted_mesh)
{
  // Skip refinement if already refined
  if (mesh_function.has_child())
  {
    dolfin_debug("MeshFunction has already been refined, returning child");
    return mesh_function.child();
  }

  dolfin_assert(mesh_function.mesh());
  const Mesh& mesh = *mesh_function.mesh();
  const std::size_t dim = mesh.topology().dim();

  // Extract parent map from data of refined mesh
  const std::vector<std::size_t>* parent = NULL;
  if (mesh_function.dim() == dim)
  {
    if (adapted_mesh->data().exists("parent_cell", dim))
      parent = &(adapted_mesh->data().array("parent_cell", dim));
  }
  else if (mesh_function.dim() == (dim - 1))
  {
    if (adapted_mesh->data().exists("parent_facet", dim - 1))
      parent = &(adapted_mesh->data().array("parent_facet", dim - 1));
  }
  else
    dolfin_not_implemented();

  // Check that parent map exists
  if (!parent)
  {
    dolfin_error("adapt.cpp",
                 "adapt mesh function",
                 "Unable to extract information about parent mesh entities");
  }

  // Use very large value as 'undefined'
  const std::size_t undefined = std::numeric_limits<std::size_t>::max();

  // Map values of mesh function into refined mesh function
  std::shared_ptr<MeshFunction<std::size_t>>
    adapted_mesh_function(new MeshFunction<std::size_t>(*adapted_mesh,
                                                        mesh_function.dim()));
  for (std::size_t i = 0; i < adapted_mesh_function->size(); i++)
  {
    const std::size_t parent_index = (*parent)[i];
    if (parent_index < mesh_function.size())
      (*adapted_mesh_function)[i] = mesh_function[parent_index];
    else
      (*adapted_mesh_function)[i] = undefined;
  }

  // Set parent / child relations
  set_parent_child(mesh_function, adapted_mesh_function);

  // Return refined mesh function
  return *adapted_mesh_function;
}
//-----------------------------------------------------------------------------
void dolfin::adapt_markers(std::vector<std::size_t>& refined_markers,
                           const Mesh& adapted_mesh,
                           const std::vector<std::size_t>& markers,
                           const Mesh& mesh)
{
  // Topological dimension
  const std::size_t D = mesh.topology().dim();

  // Check that parent maps exist
  if (!adapted_mesh.data().exists("parent_facet", D - 1))
  {
    dolfin_error("adapt.cpp",
                 "adapt markers",
                 "Unable to extract information about parent mesh entities");
  }

  // Extract parent map from data of refined mesh
  const std::vector<std::size_t>& parent_facets
    = adapted_mesh.data().array("parent_facet", D - 1);

  // Create map (parent_cell, parent_local_facet) -> [(child_cell,
  // child_local_facet), ...] for boundary facets

  std::map<std::size_t, std::vector<std::size_t>> children;
  for (FacetIterator facet(adapted_mesh); !facet.end(); ++facet)
  {
    // Ignore interior facets
    if (facet->num_entities(D) == 2)
      continue;

    // Extract index of parent facet
    const std::size_t parent_facet_index = parent_facets[facet->index()];

    children[parent_facet_index].push_back(facet->index());
  }

  // Use above map to construct refined markers
  for (auto const &marker: markers)
  {
    for (auto const &child_facet: children[marker])
      refined_markers.push_back(child_facet);
  }
}
//-----------------------------------------------------------------------------
