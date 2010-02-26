// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-09
// Last changed: 2010-02-26

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/IntersectionOperator.h>
#include <dolfin/mesh/UniformMeshRefinement.h>
#include <dolfin/mesh/LocalMeshRefinement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DofMap.h>
#include "AdaptiveObjects.h"

using namespace dolfin;

// Singleton instance
AdaptiveObjects AdaptiveObjects::objects;

// Iterators
typedef std::map<const Mesh*, std::set<FunctionSpace*> >::iterator function_space_iterator;
typedef std::map<const FunctionSpace*, std::set<Function*> >::iterator function_iterator;

// Templated function for adding edges to the forest
template <typename A, typename B>
void add_edge(std::map<const A*, std::set<B*> >& branch, const A* a, B* b)
{
  typename std::map<const A*, std::set<B*> >::iterator it = branch.find(a);

  if (it == branch.end())
  {
    std::set<B*> s;
    s.insert(b);
    branch[a] = s;
  }
  else
  {
    it->second.insert(b);
  }
}

// Templated function for removing outgoing edges for given node
template <typename A, typename B>
void remove_node(std::map<const A*, std::set<B*> >& branch, const A* a)
{
  branch.erase(a);
}

// Templated function for removing incoming edges for given node
template <typename A, typename B>
void remove_node(std::map<const A*, std::set<B*> >& branch, B* b)
{
  typename std::map<const A*, std::set<B*> >::iterator it;
  for (it = branch.begin(); it != branch.end(); ++it)
    it->second.erase(b);
}

// Templated function for refining outgoing nodes
template <typename A, typename B>
void refine_outgoing(std::map<const A*, std::set<B*> >& branch, const A* a, A& new_a)
{
  typename std::map<const A*, std::set<B*> >::iterator it = branch.find(a);
  //dolfin_debug1("Refining %d object(s)", it->second.size());
  if (it == branch.end()) return;
  for (typename std::set<B*>::iterator jt = it->second.begin(); jt != it->second.end(); ++jt)
    AdaptiveObjects::refine(*jt, new_a);
}

//-----------------------------------------------------------------------------
void AdaptiveObjects::register_object(FunctionSpace* function_space)
{
  //dolfin_debug1("Registering function space: %x", function_space);
  assert(function_space);
  add_edge(objects._function_spaces,
           &function_space->mesh(),
           function_space);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::register_object(Function* function)
{
  //dolfin_debug1("Registering function: %x", function);
  assert(function);
  add_edge(objects._functions,
           &function->function_space(),
           function);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::register_object(BoundaryCondition* boundary_condition)
{
  //dolfin_debug1("Registering boundary condition: %x", boundary_condition);
  assert(boundary_condition);
  add_edge(objects._boundary_conditions,
           &boundary_condition->function_space(),
           boundary_condition);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::deregister_object(FunctionSpace* function_space)
{
  //dolfin_debug1("Deregistering function space: %x", function_space);
  remove_node(objects._function_spaces, function_space);
  remove_node(objects._functions, function_space);
  remove_node(objects._boundary_conditions, function_space);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::deregister_object(Function* function)
{
  //dolfin_debug1("Deregistering function: %x", function);
  remove_node(objects._functions, function);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::deregister_object(BoundaryCondition* boundary_condition)
{
  //dolfin_debug1("Deregistering function space: %x", boundary_condition);
  remove_node(objects._boundary_conditions, boundary_condition);
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::refine(Mesh* mesh, MeshFunction<bool>* cell_markers)
{
  not_working_in_parallel("Mesh refinement");
  assert(mesh);

  // FIXME: This can be optimized by direct refinement (without copy)
  // FIXME: when there are no depending function spaces.

  // Type of refinement (should perhaps be a parameter)
  const bool recursive_refinement = true;

  // Create new mesh
  Mesh new_mesh;

  // Refine new mesh
  if (cell_markers)
  {
    if (recursive_refinement)
      LocalMeshRefinement::refineRecursivelyByEdgeBisection(new_mesh, *mesh, *cell_markers);
    else
      LocalMeshRefinement::refineIterativelyByEdgeBisection(new_mesh, *mesh, *cell_markers);
  }
  else
  {
    info("No cells marked for refinement, assuming uniform mesh refinement.");
    UniformMeshRefinement::refine(new_mesh, *mesh);
  }

  // Refined mesh may not be ordered
  new_mesh._ordered = false;

  // Refine all depending objects
  refine_outgoing(objects._function_spaces, mesh, new_mesh);

  // Copy data from new mesh to old
  *mesh = new_mesh;
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::refine(FunctionSpace* function_space,
                             Mesh& new_mesh)
{
  assert(function_space);

  // Create new dofmap (will be reused)
  boost::shared_ptr<ufc::dof_map> ufc_dofmap = function_space->_dofmap->_ufc_dofmap;
  boost::shared_ptr<const DofMap> new_dofmap(new DofMap(ufc_dofmap, new_mesh));

  // Create new finite element (only used temporarily)
  boost::shared_ptr<const ufc::finite_element> ufc_element = function_space->_element->_ufc_element;
  FiniteElement new_element(ufc_element);

  // Create new function space (only used temporarily)
  FunctionSpace new_function_space(reference_to_no_delete_pointer(new_mesh),
                                   reference_to_no_delete_pointer(new_element),
                                   new_dofmap);

  // Refine all depending functions and boundary conditions
  refine_outgoing(objects._functions, function_space, new_function_space);
  refine_outgoing(objects._boundary_conditions, function_space, new_function_space);

  // Copy dofmap from new function space to old
  function_space->_dofmap = new_function_space._dofmap;

  // FIXME: Might need to touch/update other data for function space here!
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::refine(Function* function,
                             FunctionSpace& new_function_space)
{
  assert(function);

  // Create new function
  Function new_function(new_function_space);

  // Interpolate from old to new function space
  new_function.interpolate(*function);

  // Copy vector from new function to old
  *function->_vector = *new_function._vector;

  // FIXME: Might need to touch/update other data for function space here!
}
//-----------------------------------------------------------------------------
void AdaptiveObjects::refine(BoundaryCondition* boundary_condition,
                             FunctionSpace& new_function_space)
{
  assert(boundary_condition);

  // Can currently only handle DirichletBC
  DirichletBC* bc = dynamic_cast<DirichletBC*>(boundary_condition);
  if (!bc)
    error("Unable to refine, automatic refinement only implemented for Dirichlet boundary conditions.");

  // Can currently only handle DirichletBC defined by SubDomain
  if (!bc->user_sub_domain)
    error("Unable to refine, automatic refinement only implemented for Dirichlet boundary conditions defined by a SubDomain.");

  // Create new boundary condition
  assert(bc->g);
  assert(bc->user_sub_domain);
  DirichletBC new_bc(new_function_space, *bc->g, *bc->user_sub_domain);

  // Copy facets from new to old bc
  bc->facets = new_bc.facets;

  // FIXME: Not reusing choice of search method for bc
}
//-----------------------------------------------------------------------------
