// Copyright (C) 2009-2013 Anders Logg
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
// First added:  2009-09-28
// Last changed: 2014-03-25

#include <string>
#include <dolfin/common/Array.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include "GenericFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction() : Variable("u", "a function")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunction::~GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericFunction::eval(Array<double>& values, const Array<double>& x,
                           const ufc::cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void GenericFunction::eval(Array<double>& values, const Array<double>& x) const
{
  dolfin_error("GenericFunction.cpp",
               "evaluate function",
               "Missing eval() function (must be overloaded)");
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x) const
{
  // Check that function is scalar
  if (value_rank() != 0)
  {
    dolfin_error("GenericFunction.cpp",
                 "evaluate function at point",
                 "Function is not scalar");
  }

  // Set up Array arguments
  double values_data[1];
  Array<double> values(1, values_data);
  const Array<double> _x(1, &x);

  // Call eval
  eval(values, _x);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x, double y) const
{
  // Check that function is scalar
  if (value_rank() != 0)
  {
    dolfin_error("GenericFunction.cpp",
                 "evaluate function at point",
                 "Function is not scalar");
  }

  // Set up Array arguments
  double values_data[1];
  Array<double> values(1, values_data);
  double _x_data[2] = { x, y };
  const Array<double> _x(2, _x_data);

  // Call eval
  eval(values, _x);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x, double y, double z) const
{
  // Check that function is scalar
  if (value_rank() != 0)
  {
    dolfin_error("GenericFunction.cpp",
                 "evaluate function at point",
                 "Function is not scalar");
  }

  // Set up Array arguments
  double values_data[1];
  Array<double> values(1, values_data);
  double _x_data[3] = { x, y, z };
  const Array<double> _x(3, _x_data);

  // Call eval
  eval(values, _x);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (const Point& p) const
{
  return (*this)(p.x(), p.y(), p.z());
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x) const
{
  // Set up Array argument
  const Array<double> _x(1, &x);

  // Call eval
  eval(values, _x);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x, double y) const
{
  // Set up Array argument
  double _x_data[2] = { x, y };
  const Array<double> _x(2, _x_data);

  // Call eval
  eval(values, _x);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x, double y, double z) const
{
  // Set up Array argument
  double _x_data[3] = { x, y, z };
  const Array<double> _x(3, _x_data);

  // Call eval
  eval(values, _x);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values, const Point& p) const
{
  (*this)(values, p.x(), p.y(), p.z());
}
//-----------------------------------------------------------------------------
std::size_t GenericFunction::value_size() const
{
  std::size_t size = 1;
  for (std::size_t i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
void GenericFunction::evaluate(double* values,
                               const double* coordinates,
                               const ufc::cell& cell) const
{
  dolfin_assert(values);
  dolfin_assert(coordinates);

  // Wrap data
  Array<double> _values(value_size(), values);
  const Array<double>
    x(cell.geometric_dimension, const_cast<double*>(coordinates));

  // Redirect to eval
  eval(_values, x, cell);
}
//-----------------------------------------------------------------------------
void GenericFunction::restrict_as_ufc_function(double* w,
                                               const FiniteElement& element,
                                               const Cell& dolfin_cell,
                                               const double* coordinate_dofs,
                                               const ufc::cell& ufc_cell) const
{
  dolfin_assert(w);

  // Evaluate dofs to get the expansion coefficients
  element.evaluate_dofs(w, *this, coordinate_dofs, ufc_cell.orientation,
                        ufc_cell);
}
//-----------------------------------------------------------------------------
