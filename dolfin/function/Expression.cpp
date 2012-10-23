// Copyright (C) 2009-2011 Anders Logg
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
// Modified by Johan Hake, 2009.
//
// First added:  2009-09-28
// Last changed: 2011-11-14

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/fem/UFCCell.h>
#include "Expression.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Expression::Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::Expression(uint dim)
{
  value_shape.resize(1);
  value_shape[0] = dim;
}
//-----------------------------------------------------------------------------
Expression::Expression(uint dim0, uint dim1)
{
  value_shape.resize(2);
  value_shape[0] = dim0;
  value_shape[1] = dim1;
}
//-----------------------------------------------------------------------------
Expression::Expression(std::vector<uint> value_shape)
  : value_shape(value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::Expression(const Expression& expression)
  : value_shape(expression.value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::~Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Expression::eval(Array<double>& values,
                      const Array<double>& x,
                      const ufc::cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void Expression::eval(Array<double>& values, const Array<double>& x) const
{
  dolfin_error("Expression.cpp",
               "evaluate expression",
               "Missing eval() function (must be overloaded)");
}
//-----------------------------------------------------------------------------
dolfin::uint Expression::value_rank() const
{
  return value_shape.size();
}
//-----------------------------------------------------------------------------
dolfin::uint Expression::value_dimension(uint i) const
{
  if (i >= value_shape.size())
  {
    dolfin_error("Expression.cpp",
                 "evaluate expression",
                 "Illegal axis %d for value dimension for value of rank %d",
                 i, value_shape.size());
  }
  return value_shape[i];
}
//-----------------------------------------------------------------------------
void Expression::restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const ufc::cell& ufc_cell) const
{
  // Restrict as UFC function (by calling eval)
  restrict_as_ufc_function(w, element, dolfin_cell, ufc_cell);
}
//-----------------------------------------------------------------------------
void Expression::compute_vertex_values(std::vector<double>& vertex_values,
                                       const Mesh& mesh) const
{
  // Local data for vertex values
  const uint size = value_size();
  Array<double> local_vertex_values(size);

  // Resize vertex_values
  vertex_values.resize(size*mesh.num_vertices());

  // Iterate over cells, overwriting values when repeatedly visiting vertices
  UFCCell ufc_cell(mesh);
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update cell data
    ufc_cell.update(*cell);

    // Iterate over cell vertices
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      // Wrap coordinate data
      const Array<double> x(mesh.geometry().dim(), const_cast<double*>(vertex->x()));

      // Evaluate at vertex
      eval(local_vertex_values, x, ufc_cell);

      // Copy to array
      for (uint i = 0; i < size; i++)
      {
        const uint global_index = i*mesh.num_vertices() + vertex->index();
        vertex_values[global_index] = local_vertex_values[i];
      }
    }
  }
}
//-----------------------------------------------------------------------------
