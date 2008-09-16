// Copyright (C) 2006-2008 Martin Sandve Alnes.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-08
// Last changed: 2008-05-08

#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/IntersectionDetector.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/UFCCell.h>
#include "UFCFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UFCFunction::UFCFunction(Mesh& mesh, const ufc::function& function, uint size)
  : GenericFunction(mesh), function(function), size(size)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UFCFunction::~UFCFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint UFCFunction::rank() const
{
  // TODO: Take rank and shape instead of size in constructor. Or better: add rank() and dim(i) to ufc::function in next version.
  return size > 1 ? 1: 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UFCFunction::dim(uint i) const
{
  // TODO: Take rank and shape instead of size in constructor. Or better: add rank() and dim(i) to ufc::function in next version.
  return size;
}
//-----------------------------------------------------------------------------
void UFCFunction::interpolate(real* values) const
{
  dolfin_assert(values);

  real * local_values = new real[size];

  // Optimization to avoid duplicated function calls
  bool * visited_vertex = new bool[mesh->numVertices()];
  for(uint i=0; i < mesh->numVertices(); i++)
    visited_vertex[i] = false;
  
  // Call function at each vertex
  for(CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    UFCCell ufc_cell(*cell);
    //ufc_cell.update(*cell);
    for(VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      uint vi = vertex->index();
      if(!visited_vertex[vi])
      {
        visited_vertex[vi] = true;

        // Evaluate function at vertex
        function.evaluate(local_values, vertex->x(), ufc_cell);

        // Copy local values to array of vertex values
        for (uint i = 0; i < size; i++)
          values[i*mesh->numVertices() + vertex->index()] = local_values[i];
      }
    }
  }
  delete [] local_values;
  delete [] visited_vertex;
}
//-----------------------------------------------------------------------------
void UFCFunction::interpolate(real* coefficients,
                              const ufc::cell& cell,
                              const FiniteElement& finite_element) const
{
  dolfin_assert(coefficients);
  
  // Compute size of value (number of entries in tensor value)
  uint fesize = 1;
  for (uint i = 0; i < finite_element.valueRank(); i++)
    fesize *= finite_element.valueDimension(i);
  dolfin_assert(fesize == size);
  
  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.spaceDimension(); i++)
    coefficients[i] = finite_element.evaluateDof(i, function, cell);
}
//-----------------------------------------------------------------------------
void UFCFunction::eval(real* values, const real* x) const
{
  dolfin_assert(values);
  dolfin_assert(x);

  // TODO: Need to find cell, then call evaluate.
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
void UFCFunction::evaluate(real* values,
                           const real* coordinates,
                           const ufc::cell& cell) const
{
  function.evaluate(values, coordinates, cell);
}
//-----------------------------------------------------------------------------
