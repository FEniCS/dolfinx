// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-02-10
// Last changed: 2011-01-31

#include <boost/shared_ptr.hpp>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/LocalMeshRefinement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/UniformMeshRefinement.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
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
dolfin::Mesh& dolfin::refine(const Mesh& mesh)
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
dolfin::Mesh& dolfin::refine(const Mesh& mesh,
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
dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space)
{
  // Refine mesh
  refine(space.mesh());

  // Refine space
  refine(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space,
                                      const MeshFunction<bool>& cell_markers)
{
  // Refine mesh
  refine(space.mesh(), cell_markers);

  // Refine space
  refine(space, space.mesh().child_shared_ptr());

  return space.child();
}
//-----------------------------------------------------------------------------
dolfin::FunctionSpace& dolfin::refine(const FunctionSpace& space,
                                      boost::shared_ptr<const Mesh> refined_mesh)
{
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
dolfin::Function& dolfin::refine(const Function& function,
                                 boost::shared_ptr<const Mesh> refined_mesh)
{
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

  // Create function on refined space and interpolate
  boost::shared_ptr<Function> refined_function(new Function(refined_space));
  refined_function->interpolate(function);

  // Set parent / child
  set_parent_child(function, refined_function);

  return *refined_function;
}
//-----------------------------------------------------------------------------
dolfin::Form& dolfin::refine(const Form& form,
                             boost::shared_ptr<const Mesh> refined_mesh)
{
  cout << "Refining form" << endl;

  // Get form data
  boost::shared_ptr<const Mesh> mesh = form.mesh_shared_ptr();
  std::vector<boost::shared_ptr<const FunctionSpace> > spaces = form.function_spaces();
  std::vector<boost::shared_ptr<const GenericFunction> > coefficients = form.coefficients();
  boost::shared_ptr<const ufc::form> ufc_form = form.ufc_form_shared_ptr();

  // FIXME: Figure out how to handle refinement of mesh

  // Refine mesh
  //boost::shared_ptr<const Mesh> refined_mesh;
  //if (mesh != 0)
  //{
  //  refine(mesh);
  //  refined_mesh = mesh.child_shared_ptr();
  //}

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
    }

    // If not, just reuse the same coefficient
    else
    {
      refined_coefficients.push_back(coefficients[i]);
    }
  }

  /// Create form (constructor used from Python interface)
  boost::shared_ptr<Form> refined_form(new Form(ufc_form,
                                                refined_spaces,
                                                refined_coefficients));

  // Set parent / child
  set_parent_child(form, refined_form);

  return *refined_form;
}
//-----------------------------------------------------------------------------
