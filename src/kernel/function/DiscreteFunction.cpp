// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-04-02
// Last changed: 2007-05-02

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Form.h>
#include <dolfin/DofMap.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/UFCCell.h>
#include <dolfin/ElementLibrary.h>
#include <dolfin/SubFunction.h>
#include <dolfin/SubSystem.h>
#include <dolfin/DiscreteFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, Vector& x, const Form& form, uint i)
  : GenericFunction(mesh),
    x(&x), finite_element(0), dof_map(0), ufc_dof_map(0), dofs(0),
    local_mesh(false), local_vector(false)
{
  // Check argument
  const uint num_arguments = form.form().rank() + form.form().num_coefficients();
  if (i >= num_arguments)
  {
    dolfin_error("Illegal function index %d. Form only has %d arguments.",
                  i, num_arguments);
  }

  // Create finite element
  finite_element = form.form().create_finite_element(i);

  // Create dof map
  ufc_dof_map = form.form().create_dof_map(i);
  dof_map = new DofMap(*ufc_dof_map, mesh);

  // Initialize vector
  if (x.size() != dof_map->global_dimension())
    x.init(dof_map->global_dimension());

  // Initialize local array for mapping of dofs
  dofs = new uint[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
    dofs[i] = 0;
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, Vector& x,
                                   std::string finite_element_signature,
                                   std::string dof_map_signature)
  : GenericFunction(mesh),
    x(&x), finite_element(0), dof_map(0), ufc_dof_map(0), dofs(0),
    local_mesh(false), local_vector(false)
{
  // Create finite element
  finite_element = ElementLibrary::create_finite_element(finite_element_signature);
  if (!finite_element)
  {
    dolfin_error("Unable to find finite element in library: \"%s\".",
                  finite_element_signature.c_str());
  }

  // Create dof map
  ufc_dof_map = ElementLibrary::create_dof_map(dof_map_signature);
  if (!ufc_dof_map)
  {
    dolfin_error("Unable to find dof map in library: \"%s\".",
                  dof_map_signature.c_str());
  }
  dof_map = new DofMap(*ufc_dof_map, mesh);

  // Check size of vector
  if (x.size() != dof_map->global_dimension())
    dolfin_error("Size of vector does not match global dimension of finite element space.");

  // Initialize local array for mapping of dofs
  dofs = new uint[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
    dofs[i] = 0;

  // Assume responsibility for mesh and vector
  local_mesh = &mesh;
  local_vector = &x;
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(SubFunction& sub_function)
  : GenericFunction(sub_function.f->mesh),
    x(0), finite_element(0), dof_map(0), ufc_dof_map(0), dofs(0),
    local_mesh(false), local_vector(false)
{
  // Create sub system
  SubSystem sub_system(sub_function.i);

  // Extract sub element
  finite_element = sub_system.extractFiniteElement(*sub_function.f->finite_element);

  // Extract sub dof map and offset
  uint offset = 0;
  ufc_dof_map = sub_system.extractDofMap(*sub_function.f->ufc_dof_map, mesh, offset);
  dof_map = new DofMap(*ufc_dof_map, mesh);

  // Create vector of dofs and copy values
  const uint n = dof_map->global_dimension();
  x = new Vector(n);
  real* values = new real[n];
  uint* get_rows = new uint[n];
  uint* set_rows = new uint[n];
  for (uint i = 0; i < n; i++)
  {
    get_rows[i] = offset + i;
    set_rows[i] = i;
  }
  sub_function.f->x->get(values, n, get_rows);
  x->set(values, n, set_rows);
  delete [] values;
  delete [] get_rows;
  delete [] set_rows;

  // Initialize local array for mapping of dofs
  dofs = new uint[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
    dofs[i] = 0;

  // Assume responsibility for vector
  local_vector = x;
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(const DiscreteFunction& f)
  : GenericFunction(f.mesh),
    x(0), finite_element(0), dof_map(0), ufc_dof_map(0), dofs(0),
    local_mesh(false), local_vector(false)
{
  cout << "Copy constructor for discrete function" << endl;

  // Create finite element
  finite_element = ElementLibrary::create_finite_element(f.finite_element->signature());
  if (!finite_element)
  {
    dolfin_error("Unable to find finite element in library: \"%s\".",
                  f.finite_element->signature());
  }

  // Create dof map
  ufc_dof_map = ElementLibrary::create_dof_map(f.dof_map->signature());
  if (!ufc_dof_map)
  {
    dolfin_error("Unable to find dof map in library: \"%s\".",
                  f.dof_map->signature());
  }
  dof_map = new DofMap(*ufc_dof_map, mesh);

  // Create vector and copy values
  x = new Vector(dof_map->global_dimension());
  *x = *f.x;

  // Initialize local array for mapping of dofs
  dofs = new uint[dof_map->local_dimension()];
  for (uint i = 0; i < dof_map->local_dimension(); i++)
    dofs[i] = 0;

  // Assume responsibility for vector
  local_vector = x;
}
//-----------------------------------------------------------------------------
DiscreteFunction::~DiscreteFunction()
{
  if (finite_element)
    delete finite_element;
      
  if (dof_map)
    delete dof_map;

  if (ufc_dof_map)
    delete ufc_dof_map;

  if (dofs)
    delete [] dofs;

  if (local_mesh)
    delete local_mesh;

  if (local_vector)
    delete local_vector;
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::rank() const
{
  dolfin_assert(finite_element);
  return finite_element->value_rank();
}
//-----------------------------------------------------------------------------
dolfin::uint DiscreteFunction::dim(uint i) const
{
  dolfin_assert(finite_element);
  return finite_element->value_dimension(i);
}
//-----------------------------------------------------------------------------
const DiscreteFunction& DiscreteFunction::operator= (const DiscreteFunction& f)
{
  cout << "Assignment in discrete function" << endl;

  // Check that data matches
  if (strcmp(finite_element->signature(), f.finite_element->signature()) != 0 ||
       strcmp(dof_map->signature(), f.dof_map->signature()) != 0 ||
      x->size() != f.x->size())
  {
    dolfin_error("Assignment of discrete function failed. Finite element spaces or dimensions don't match.");
  }

  // Copy vector
  *x = *f.x;

  return *this;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* values)
{
  dolfin_assert(values);
  dolfin_assert(finite_element);
  dolfin_assert(dof_map);
  dolfin_assert(dofs);
  
  // Compute size of value (number of entries in tensor value)
  uint size = 1;
  for (uint i = 0; i < finite_element->value_rank(); i++)
    size *= finite_element->value_dimension(i);

  // Local data for interpolation on each cell
  CellIterator cell(mesh);
  UFCCell ufc_cell(*cell);
  const uint num_cell_vertices = mesh.type().numVertices(mesh.topology().dim());
  real* vertex_values = new real[size*num_cell_vertices];
  real* dof_values = new real[finite_element->space_dimension()];

  // Interpolate vertex values on each cell and pick the last value
  // if two or more cells disagree on the vertex values
  for (; !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs
    dof_map->tabulate_dofs(dofs, ufc_cell);
    
    // Pick values from global vector
    x->get(dof_values, dof_map->local_dimension(), dofs);

    // Interpolate values at the vertices
    finite_element->interpolate_vertex_values(vertex_values, dof_values, ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
      for (uint i = 0; i < size; ++i)
        values[i*mesh.numVertices() + vertex->index()] = vertex_values[i*num_cell_vertices + vertex.pos()];
  }

  // Delete local data
  delete [] vertex_values;
  delete [] dof_values;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element)
{
  dolfin_assert(coefficients);
  dolfin_assert(this->finite_element);
  dolfin_assert(this->dof_map);
  dolfin_assert(this->dofs);

  // FIXME: Better test here, compare against the local element

  // Check dimension
  if (finite_element.space_dimension() != dof_map->local_dimension())
    dolfin_error("Finite element does not match for interpolation of discrete function.");

  // Tabulate dofs
  dof_map->tabulate_dofs(dofs, cell);
  
  // Pick values from global vector
  x->get(coefficients, dof_map->local_dimension(), dofs);
}
//-----------------------------------------------------------------------------
