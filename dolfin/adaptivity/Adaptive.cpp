// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-02
// Last changed: 2009-11-09

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/UniformMeshRefinement.h>
#include <dolfin/mesh/LocalMeshRefinement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "Adaptive.h"

using namespace dolfin;

typedef std::set<FunctionSpace*>::iterator function_space_iterator;
typedef std::set<Function*>::iterator function_iterator;
typedef std::set<BoundaryCondition*>::iterator boundary_condition_iterator;

//-----------------------------------------------------------------------------
void Adaptive::register_object(FunctionSpace* function_space) const
{
  std::cout << "registering V = " << function_space << std::endl;

  _function_spaces.insert(function_space);
}
//-----------------------------------------------------------------------------
void Adaptive::register_object(Function* function) const
{
  std::cout << "registering v = " << function << " for V = " << this << std::endl;

  _functions.insert(function);
}
//-----------------------------------------------------------------------------
void Adaptive::register_object(BoundaryCondition* boundary_condition) const
{
  _boundary_conditions.insert(boundary_condition);
}
//-----------------------------------------------------------------------------
void Adaptive::deregister_object(FunctionSpace* function_space) const
{
  _function_spaces.erase(function_space);
}
//-----------------------------------------------------------------------------
void Adaptive::deregister_object(Function* function) const
{
  _functions.erase(function);
}
//-----------------------------------------------------------------------------
void Adaptive::deregister_object(BoundaryCondition* boundary_condition) const
{
  _boundary_conditions.erase(boundary_condition);
}
//-----------------------------------------------------------------------------
void Adaptive::refine_function_spaces(const Mesh& new_mesh)
{
  if (_function_spaces.size() > 0)
    info("Refining %d function space(s).", _function_spaces.size());

  for (function_space_iterator it = _function_spaces.begin();
       it != _function_spaces.end(); ++it)
    refine_function_space(**it, new_mesh);
}
//-----------------------------------------------------------------------------
void Adaptive::refine_functions(const FunctionSpace& new_function_space)
{
  if (_functions.size() > 0)
    info("Refining %d function(s).", _functions.size());

  for (function_iterator it = _functions.begin();
       it != _functions.end(); ++it)
    refine_function(**it, new_function_space);
}
//-----------------------------------------------------------------------------
void Adaptive::refine_boundary_conditions(const FunctionSpace& new_function_space)
{
  if (_boundary_conditions.size() > 0)
    info("Refining %d boundary condition(s).", _boundary_conditions.size());

  for (boundary_condition_iterator it = _boundary_conditions.begin();
       it != _boundary_conditions.end(); ++it)
    refine_boundary_condition(**it, new_function_space);
}
//-----------------------------------------------------------------------------
void Adaptive::refine_mesh(Mesh& mesh,
                           MeshFunction<bool>* cell_markers)
{
  not_working_in_parallel("Mesh refinement");

  // FIXME: This can be optimized by direct refinement (without copy)
  // FIXME: when there are no depending function spaces.

  // Type of refinement (should perhaps be a parameter)
  const bool recursive_refinement = true;

  // Create new mesh (copy of old mesh)
  Mesh new_mesh(mesh);

  // Refine new mesh
  if (cell_markers)
  {
    if (recursive_refinement)
      LocalMeshRefinement::refineRecursivelyByEdgeBisection(new_mesh, *cell_markers);
    else
      LocalMeshRefinement::refineIterativelyByEdgeBisection(new_mesh, *cell_markers);
  }
  else
  {
    info("No cells marked for refinement, assuming uniform mesh refinement.");
    UniformMeshRefinement::refine(new_mesh);
  }

  // Refined mesh may not be ordered
  new_mesh._ordered = false;

  // Refine all depending function spaces
  refine_function_spaces(new_mesh);

  // Copy data from new mesh to old
  mesh = new_mesh;
}
//-----------------------------------------------------------------------------
void Adaptive::refine_function_space(FunctionSpace& function_space,
                                     const Mesh& new_mesh)
{
  cout << "Refining function space " << endl;

  std::cout << "refine V: " << this << std::endl;

  // Create new dofmap (will be reused)
  boost::shared_ptr<ufc::dof_map> ufc_dofmap = function_space._dofmap->_ufc_dofmap;
  boost::shared_ptr<const DofMap> new_dofmap(new DofMap(ufc_dofmap, new_mesh));

  // Create new finite element (only used temporarily)
  boost::shared_ptr<const ufc::finite_element> ufc_element = function_space._element->_ufc_element;
  FiniteElement new_element(ufc_element);

  // Create new function space (only used temporarily)
  FunctionSpace new_function_space(reference_to_no_delete_pointer(new_mesh),
                                   reference_to_no_delete_pointer(new_element),
                                   new_dofmap);

  cout << "Number of functions: " << _functions.size() << endl;

  // Refine all depending functions and boundary conditions
  refine_functions(new_function_space);
  refine_boundary_conditions(new_function_space);

  // Copy dofmap from new function space to old
  function_space._dofmap = new_function_space._dofmap;
}
//-----------------------------------------------------------------------------
void Adaptive::refine_function(Function& function,
                               const FunctionSpace& new_function_space)
{
  // Create new function
  Function new_function(new_function_space);

  // Interpolate from old to new function space
  new_function.interpolate(function);

  // Copy vector from new function to old
  function._vector->resize(new_function_space.dim());
  *function._vector = *new_function._vector;

  cout << "size after refine_function: " << function._vector->size() << endl;
}
//-----------------------------------------------------------------------------
void Adaptive::refine_boundary_condition(BoundaryCondition& boundary_condition,
                                         const FunctionSpace& new_function_space)
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
