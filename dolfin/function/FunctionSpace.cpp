// Copyright (C) 2008 Anders Logg (and others?).
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-25

#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(Mesh& mesh, const FiniteElement &element, const DofMap& dofmap)
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
FunctionSpace::~FunctionSpace()
{
  delete intersection_detector;
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
void FunctionSpace::eval(real* values, const real* p, const GenericVector& x) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Initialize intersection detector if not done before
  if (!intersection_detector)
    intersection_detector = new IntersectionDetector(*_mesh);

  // Find the cell that contains p
  Point point(_mesh->geometry().dim(), p);
  Array<uint> cells;
  intersection_detector->overlap(point, cells);
  if (cells.size() < 1)
    error("Unable to evaluate function at given point (not inside domain).");
  else if (cells.size() > 1)
    warning("Point belongs to more than one cell, picking first.");
  Cell cell(*_mesh, cells[0]);
  UFCCell ufc_cell(cell);
  
  // Tabulate dofs
  _dofmap->tabulate_dofs(scratch.dofs, ufc_cell);
  
  // Pick values from vector of dofs
  x.get(scratch.coefficients, _dofmap->local_dimension(), scratch.dofs);

  // Compute linear combination
  for (uint j = 0; j < scratch.size; j++)
    values[j] = 0.0;
  for (uint i = 0; i < _element->space_dimension(); i++)
  {
    _element->evaluate_basis(i, scratch.values, p, ufc_cell);
    for (uint j = 0; j < scratch.size; j++)
      values[j] += scratch.coefficients[i] * scratch.values[j];
  }
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::Scratch(const FiniteElement& element)
  : size(0), dofs(0), coefficients(0), values(0)
{
  // Compute size of value (number of entries in tensor value)
  size = 1;
  for (uint i = 0; i < element.value_rank(); i++)
    size *= element.value_dimension(i);

  // Initialize local array for mapping of dofs
  dofs = new uint[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    dofs[i] = 0;

  // Initialize local array for expansion coefficients
  coefficients = new real[element.space_dimension()];
  for (uint i = 0; i < element.space_dimension(); i++)
    coefficients[i] = 0.0;

  // Initialize local array for values
  values = new real[size];
  for (uint i = 0; i < size; i++)
    values[i] = 0.0;  
}
//-----------------------------------------------------------------------------
FunctionSpace::Scratch::~Scratch()
{
  delete [] dofs;
  delete [] coefficients;
  delete [] values;
}
//-----------------------------------------------------------------------------
