// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
// Modified by Dag Lindbo, 2008.
// Modified by Kristen Kaasbjerg, 2008.
//
// First added:  2007-04-02
// Last changed: 2008-03-17

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/UFCMesh.h>
#include <dolfin/fem/UFCCell.h>
#include <dolfin/fem/SubSystem.h>
#include <dolfin/elements/ElementLibrary.h>
#include "SubFunction.h"
#include "DiscreteFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, GenericVector& x, Form& form, uint i)
  : GenericFunction(mesh),
    x(&x), finite_element(0), dof_map(0),
    local_vector(0), local_dof_map(0), intersection_detector(0), scratch(0)
{
  // Update dof maps
  form.updateDofMaps(mesh);
  dof_map = &form.dofMaps()[i];

  // Initialise function
  init(mesh, x, form.form(), i);
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, GenericVector& x, DofMap& dof_map,
                                   const ufc::form& form, uint i)
  : GenericFunction(mesh),
    x(&x), finite_element(0), dof_map(&dof_map),
    local_vector(0), local_dof_map(0), intersection_detector(0), scratch(0)
{
  init(mesh, x, form, i);
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(Mesh& mesh, GenericVector& x,
                                   std::string finite_element_signature,
                                   std::string dof_map_signature)
  : GenericFunction(mesh),
    x(&x), finite_element(0), dof_map(0),
    local_vector(0), local_dof_map(0), intersection_detector(0), scratch(0)
{
  // Create finite element
  finite_element = ElementLibrary::create_finite_element(finite_element_signature);
  if (!finite_element)
  {
    error("Unable to find finite element in library: \"%s\".",
                  finite_element_signature.c_str());
  }

  // Create dof map from signature
  dof_map = new DofMap(dof_map_signature, mesh);

  // Check size of vector
  if (x.size() != dof_map->global_dimension())
    error("Size of vector does not match global dimension of finite element space.");

  // Assume responsibility for data
  local_dof_map = dof_map;

  // Initialize scratch space
  scratch = new Scratch(*finite_element);
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(SubFunction& sub_function)
  : GenericFunction(sub_function.f->mesh),
    x(0), finite_element(0), dof_map(0),
    local_vector(0), local_dof_map(0), intersection_detector(0), scratch(0)
{
  // Create sub system
  SubSystem sub_system(sub_function.i);

  // Extract sub element
  finite_element = sub_system.extractFiniteElement(*sub_function.f->finite_element);

  // Extract sub dof map and offset
  uint offset = 0;
  dof_map = sub_function.f->dof_map->extractDofMap(sub_system.array(), offset);

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
  x->apply();
  delete [] values;
  delete [] get_rows;
  delete [] set_rows;

  // Assume responsibility for vector and dof map
  local_vector = x;
  local_dof_map = dof_map;

  // Initialize scratch space
  scratch = new Scratch(*finite_element);
}
//-----------------------------------------------------------------------------
DiscreteFunction::DiscreteFunction(const DiscreteFunction& f)
  : GenericFunction(f.mesh),
    local_vector(0), local_dof_map(0), intersection_detector(0), scratch(0)
{
  cout << "Copy constructor for discrete function" << endl;

  // FIXME: Why don't we just copy the finite_element?
  // Create finite element
  finite_element = ElementLibrary::create_finite_element(f.finite_element->signature());
  if (!finite_element)
    error("Unable to find finite element in library: \"%s\".",
                  f.finite_element->signature());

  // Create dof map
  dof_map = new DofMap(f.dof_map->signature(), mesh); 

  // Create vector and copy values
  x  = new Vector(dof_map->global_dimension());
  *x = *f.x;

  // Assume responsibility for vector and dof map
  local_vector = x;
  local_dof_map = dof_map;

  // Initialize scratch space
  scratch = new Scratch(*finite_element);
}
//-----------------------------------------------------------------------------
DiscreteFunction::~DiscreteFunction()
{
  if (finite_element)
    delete finite_element;
      
  if (local_vector)
    delete local_vector;

  if (local_dof_map)
    delete local_dof_map;
  
  if (intersection_detector)
    delete intersection_detector;

  if (scratch)
    delete scratch;
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
dolfin::uint DiscreteFunction::numSubFunctions() const
{
  dolfin_assert(finite_element);
  return finite_element->num_sub_elements();
}
//-----------------------------------------------------------------------------
const DiscreteFunction& DiscreteFunction::operator= (const DiscreteFunction& f)
{
  cout << "Assignment in discrete function" << endl;

  // Check that data matches
  if (strcmp(finite_element->signature(), f.finite_element->signature()) != 0 ||
       strcmp(dof_map->signature(), f.dof_map->signature()) != 0              ||
      x->size() != f.x->size())
  {
    error("Assignment of discrete function failed. Finite element spaces or dimensions don't match.");
  }

  // Copy vector
  *x = *f.x;

  return *this;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* values) const
{
  dolfin_assert(values);
  dolfin_assert(finite_element);
  dolfin_assert(dof_map);
  dolfin_assert(scratch);
  
  // Local data for interpolation on each cell
  CellIterator cell(mesh);
  UFCCell ufc_cell(*cell);
  const uint num_cell_vertices = mesh.type().numVertices(mesh.topology().dim());
  real* vertex_values = new real[scratch->size*num_cell_vertices];

  // Interpolate vertex values on each cell and pick the last value
  // if two or more cells disagree on the vertex values
  for (; !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs
    dof_map->tabulate_dofs(scratch->dofs, ufc_cell);
    
    // Pick values from global vector
    x->get(scratch->coefficients, dof_map->local_dimension(), scratch->dofs);

    // Interpolate values at the vertices
    finite_element->interpolate_vertex_values(vertex_values, scratch->coefficients, ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
      for (uint i = 0; i < scratch->size; ++i)
        values[i*mesh.numVertices() + vertex->index()] = vertex_values[i*num_cell_vertices + vertex.pos()];
  }

  // Delete local data
  delete [] vertex_values;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::interpolate(real* coefficients,
                                   const ufc::cell& cell,
                                   const ufc::finite_element& finite_element) const
{
  dolfin_assert(coefficients);
  dolfin_assert(this->finite_element);
  dolfin_assert(this->dof_map);
  dolfin_assert(scratch);

  // FIXME: Better test here, compare against the local element

  // Check dimension
  if (finite_element.space_dimension() != dof_map->local_dimension())
    error("Finite element does not match for interpolation of discrete function.");

  // Tabulate dofs
  dof_map->tabulate_dofs(scratch->dofs, cell);
  
  // Pick values from global vector
  x->get(coefficients, dof_map->local_dimension(), scratch->dofs);
}
//-----------------------------------------------------------------------------
void DiscreteFunction::eval(real* values, const real* x) const
{
  dolfin_assert(scratch);

  // Initialize intersection detector if not done before
  if (!intersection_detector)
    intersection_detector = new IntersectionDetector(mesh);

  // Find the cell that contains x
  const uint gdim = mesh.geometry().dim();
  if (gdim > 3)
    error("Sorry, point evaluation of functions not implemented for meshes of dimension %d.", gdim);
  Point p;
  for (uint i = 0; i < gdim; i++)
    p[i] = x[i];
  Array<uint> cells;
  intersection_detector->overlap(p, cells);
  if (cells.size() < 1)
    error("Unable to evaluate function at given point (not inside domain).");
  Cell cell(mesh, cells[0]);
  UFCCell ufc_cell(cell);
  
  // Get expansion coefficients on cell
  this->interpolate(scratch->coefficients, ufc_cell, *finite_element);

  // Compute linear combination
  for (uint j = 0; j < scratch->size; j++)
    values[j] = 0.0;
  for (uint i = 0; i < finite_element->space_dimension(); i++)
  {
    finite_element->evaluate_basis(i, scratch->values, x, ufc_cell);
    for (uint j = 0; j < scratch->size; j++)
      values[j] += scratch->coefficients[i] * scratch->values[j];
  }
}
//-----------------------------------------------------------------------------
GenericVector& DiscreteFunction::vector() const
{
  if( !x )
    error("Vector associated with DiscreteFunction has not been initialised.");

  return *x;
}
//-----------------------------------------------------------------------------
void DiscreteFunction::init(Mesh& mesh, GenericVector& x, const ufc::form& form, uint i)
{
  // Check argument
  const uint num_arguments = form.rank() + form.num_coefficients();
  if (i >= num_arguments)
    error("Illegal function index %d. Form only has %d arguments.",
                  i, num_arguments);

  // Create finite element
  finite_element = form.create_finite_element(i);

  // Initialize vector
  if (x.size() != dof_map->global_dimension())
    x.init(dof_map->global_dimension());

  // Initialize scratch space
  if (!scratch)
    scratch = new Scratch(*finite_element);
}
//-----------------------------------------------------------------------------
DiscreteFunction::Scratch::Scratch(ufc::finite_element& finite_element)
  : size(0), dofs(0), coefficients(0), values(0)
{
  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < finite_element.value_rank(); i++)
    size *= finite_element.value_dimension(i);

  // Initialize local array for mapping of dofs
  dofs = new uint[finite_element.space_dimension()];
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    dofs[i] = 0;

  // Initialize local array for expansion coefficients
  coefficients = new real[finite_element.space_dimension()];
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = 0.0;

  // Initialize local array for values
  values = new real[size];
  for (uint i = 0; i < size; i++)
    values[i] = 0.0;  
}
//-----------------------------------------------------------------------------
DiscreteFunction::Scratch::~Scratch()
{
  if (dofs)
    delete [] dofs;

  if (coefficients)
    delete [] coefficients;

  if (values)
    delete [] values;
}
//-----------------------------------------------------------------------------
