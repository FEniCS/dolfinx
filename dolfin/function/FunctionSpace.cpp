// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristoffer Selim, 2008.
//
// First added:  2008-09-11
// Last changed: 2008-10-14

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "Function.h"
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(Mesh& mesh,
                             const FiniteElement &element,
                             const DofMap& dofmap)
  : _mesh(&mesh, NoDeleter<Mesh>()),
    _element(&element, NoDeleter<const FiniteElement>()),
    _dofmap(&dofmap, NoDeleter<const DofMap>()),
    scratch(element), intersection_detector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::tr1::shared_ptr<Mesh> mesh,
                             std::tr1::shared_ptr<const FiniteElement> element,
                             std::tr1::shared_ptr<const DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap),
    scratch(*element), intersection_detector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(const FunctionSpace& V)
  : _mesh(static_cast<Mesh*>(0)),
    _element(static_cast<FiniteElement*>(0)),
    _dofmap(static_cast<DofMap*>(0)),
    scratch(), intersection_detector(0)
{
  *this = V;
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  delete intersection_detector;
}
//-----------------------------------------------------------------------------
const FunctionSpace& FunctionSpace::operator= (const FunctionSpace& V)
{
  // Assign data (will be shared)
  _mesh    = V._mesh;
  _element = V._element;
  _dofmap  = V._dofmap;

  // Reinitialize scratch space and intersection detector
  scratch.init(*_element);
  delete intersection_detector;
  intersection_detector = 0;

  return *this;
}
//-----------------------------------------------------------------------------
Mesh& FunctionSpace::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const Mesh& FunctionSpace::mesh() const
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const FiniteElement& FunctionSpace::element() const
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
const DofMap& FunctionSpace::dofmap() const
{
  dolfin_assert(_dofmap);
  return *_dofmap;
}
//-----------------------------------------------------------------------------
void FunctionSpace::eval(double* values,
                         const double* x,
                         const Function& v) const
{
  dolfin_assert(values);
  dolfin_assert(x);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Initialize intersection detector if not done before
  if (!intersection_detector)
    intersection_detector = new IntersectionDetector(*_mesh);

  // Find the cell that contains x
  Point point(_mesh->geometry().dim(), x);
  Array<uint> cells;
  intersection_detector->intersection(point, cells);
  if (cells.size() < 1)
    error("Unable to evaluate function at given point (not inside domain).");
  else if (cells.size() > 1)
    warning("Point belongs to more than one cell, picking first.");
  Cell cell(*_mesh, cells[0]);
  UFCCell ufc_cell(cell);

  // Tabulate dofs
  _dofmap->tabulate_dofs(scratch.dofs, ufc_cell);

  // Pick values from vector of dofs
  v.vector().get(scratch.coefficients, _dofmap->local_dimension(), scratch.dofs);

  // Compute linear combination
  for (uint j = 0; j < scratch.size; j++) 
    values[j] = 0.0;
  for (uint i = 0; i < _element->space_dimension(); i++)
  {
    _element->evaluate_basis(i, scratch.values, x, ufc_cell);
    for (uint j = 0; j < scratch.size; j++)
      values[j] += scratch.coefficients[i] * scratch.values[j];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(GenericVector& coefficients,
                                const Function& v) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);
  dolfin_assert(&v.function_space().mesh() == &mesh());

  // Initialize vector of coefficients
  coefficients.resize(_dofmap->global_dimension());
  coefficients.zero();

  // Iterate over mesh and interpolate on each cell
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Interpolate on cell
    v.interpolate(scratch.coefficients, ufc_cell);

    // Tabulate dofs
    _dofmap->tabulate_dofs(scratch.dofs, ufc_cell);

    // Copy dofs to vector
    coefficients.set(scratch.coefficients, _dofmap->local_dimension(), scratch.dofs);
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(double* coefficients,
                                const ufc::cell& ufc_cell,
                                const Function& v) const
{
  dolfin_assert(coefficients);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Tabulate dofs
  _dofmap->tabulate_dofs(scratch.dofs, ufc_cell);

  // Pick values from global vector
  v.vector().get(coefficients, _dofmap->local_dimension(), scratch.dofs);
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(double* vertex_values,
                                const Function& v) const
{
  dolfin_assert(vertex_values);
  dolfin_assert(v.in(*this));
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Local data for interpolation on each cell
  const uint num_cell_vertices = _mesh->type().numVertices(_mesh->topology().dim());
  double* local_vertex_values = new double[scratch.size*num_cell_vertices];

  // Interpolate vertex values on each cell (using latest value if not continuous)
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Tabulate dofs
    _dofmap->tabulate_dofs(scratch.dofs, ufc_cell);

    // Pick values from global vector
    v.vector().get(scratch.coefficients, _dofmap->local_dimension(), scratch.dofs);

    // Interpolate values at the vertices
    _element->interpolate_vertex_values(local_vertex_values, scratch.coefficients, ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      for (uint i = 0; i < scratch.size; ++i)
      {
        const uint local_index  = vertex.pos()*scratch.size + i;
        const uint global_index = i*_mesh->numVertices() + vertex->index();
        vertex_values[global_index] = local_vertex_values[local_index];
      }
    }
  }

  // Delete local data
  delete [] local_vertex_values;
}
//-----------------------------------------------------------------------------
FunctionSpace* FunctionSpace::extract_sub_space(const Array<uint>& sub_system) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Extract sub element
  std::tr1::shared_ptr<FiniteElement> element(_element->extract_sub_element(sub_system));

  // Extract sub dofmap and offset
  uint offset = 0;
  std::tr1::shared_ptr<DofMap> dofmap(_dofmap->extract_sub_dofmap(sub_system, offset));
  
  return new FunctionSpace(_mesh, element, dofmap);
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::Scratch(const FiniteElement& element)
  : size(0), dofs(0), coefficients(0), values(0)
{
  init(element);
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::Scratch()
  : size(0), dofs(0), coefficients(0), values(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::~Scratch()
{
  delete [] dofs;
  delete [] coefficients;
  delete [] values;
}
//-----------------------------------------------------------------------------
void FunctionSpace::Scratch::init(const FiniteElement& element)
{
  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    size *= element.value_dimension(i);

  // Initialize local array for mapping of dofs
  delete [] dofs;
  dofs = new uint[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    dofs[i] = 0;

  // Initialize local array for expansion coefficients
  delete coefficients;
  coefficients = new double[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    coefficients[i] = 0.0;

  // Initialize local array for values
  delete values;
  values = new double[size];
  for (uint i = 0; i < size; i++)
    values[i] = 0.0;
}
//-----------------------------------------------------------------------------
