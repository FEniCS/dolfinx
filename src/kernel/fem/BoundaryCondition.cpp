// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-11
// Last changed: 2007-08-18

#include <dolfin/Form.h>
#include <dolfin/SubSystem.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::LocalData(const ufc::form& form,
                                        Mesh& mesh,
                                        const SubSystem& sub_system)
  : finite_element(0), dof_map(0), offset(0),
    w(0), cell_dofs(0), values(0), x_values(0), facet_dofs(0), rows(0)
{
  // FIXME: Change behaviour of num_sub_elements() in FFC (return 0 when
  // FIXME: there are no nested elements

  // Check arity of form
  if (form.rank() != 2)
    error("Form must be bilinear for application of boundary conditions.");

  // Create finite element and dof map for solution (second argument of form)
  finite_element = form.create_finite_element(1);
  dof_map = form.create_dof_map(1);
  
  // Extract sub element and sub dof map if we have a sub system
  if (sub_system.depth() > 0)
  {
    // Finite element
    ufc::finite_element* sub_finite_element = sub_system.extractFiniteElement(*finite_element);
    delete finite_element;
    finite_element = sub_finite_element;

    // Dof map
    ufc::dof_map* sub_dof_map = sub_system.extractDofMap(*dof_map, mesh, offset);
    delete dof_map;
    dof_map = sub_dof_map;
  }

  // Create local data used to set boundary conditions
  w = new real[finite_element->space_dimension()];
  cell_dofs = new uint[finite_element->space_dimension()];
  for (uint i = 0; i < finite_element->space_dimension(); i++)
  {
    w[i] = 0.0;
    cell_dofs[i] = 0;
  }
  values = new real[dof_map->num_facet_dofs()];
  x_values = new real[dof_map->num_facet_dofs()];
  facet_dofs = new uint[dof_map->num_facet_dofs()];
  rows = new uint[dof_map->num_facet_dofs()];
  for (uint i = 0; i < dof_map->num_facet_dofs(); i++)
  {
    values[i] = 0.0;
    x_values[i] = 0.0;
    facet_dofs[i] = 0;
    rows[i] = 0;
  }

  // Create local coordinate data
  coordinates = new real*[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
  {
    coordinates[i] = new real[mesh.geometry().dim()];
    for (uint j = 0; j < mesh.geometry().dim(); j++)
      coordinates[i][j] = 0.0;
  }
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::~LocalData()
{
  if (coordinates)
  {
    for (uint i = 0; i < dof_map->local_dimension(); i++)
      delete [] coordinates[i];
    delete [] coordinates;
  }

  if (finite_element)
    delete finite_element;

  if (dof_map)
    delete dof_map;

  if (w)
    delete [] w;

  if (cell_dofs)
    delete [] cell_dofs;

  if (values)
    delete [] values;

  if (x_values)
    delete [] x_values;

  if (facet_dofs)
    delete [] facet_dofs;

  if (rows)
    delete [] rows;
}
//-----------------------------------------------------------------------------
