// Copyright (C) 2009 Anders Logg
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
// Last changed: 2011-01-19

#include <string>
#include <dolfin/fem/FiniteElement.h>
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
  error("Missing eval() function (must be overloaded).");
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x)
{
  // Check that function is scalar
  if (value_rank() != 0)
    error("Function is not scalar.");

  // Set up Array arguments
  Array<double> values(1);
  Array<double> xx(1);
  xx[0] = x;

  // Call eval
  eval(values, xx);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x, double y)
{
  // Check that function is scalar
  if (value_rank() != 0)
    error("Function is not scalar.");

  // Set up Array arguments
  Array<double> values(1);
  Array<double> xx(2);
  xx[0] = x;
  xx[1] = y;

  // Call eval
  eval(values, xx);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (double x, double y, double z)
{
  // Check that function is scalar
  if (value_rank() != 0)
    error("Function is not scalar.");

  // Set up Array arguments
  Array<double> values(1);
  Array<double> xx(3);
  xx[0] = x;
  xx[1] = y;
  xx[2] = z;

  // Call eval
  eval(values, xx);

  // Return value
  return values[0];
}
//-----------------------------------------------------------------------------
double GenericFunction::operator() (const Point& p)
{
  return (*this)(p.x(), p.y(), p.z());
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x)
{
  // Set up Array argument
  Array<double> xx(1);
  xx[0] = x;

  // Call eval
  eval(values, xx);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x, double y)
{
  // Set up Array argument
  Array<double> xx(2);
  xx[0] = x;
  xx[1] = y;

  // Call eval
  eval(values, xx);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values,
                                  double x, double y, double z)
{
  // Set up Array argument
  Array<double> xx(3);
  xx[0] = x;
  xx[1] = y;
  xx[2] = z;

  // Call eval
  eval(values, xx);
}
//-----------------------------------------------------------------------------
void GenericFunction::operator() (Array<double>& values, const Point& p)
{
  (*this)(values, p.x(), p.y(), p.z());
}
//-----------------------------------------------------------------------------
dolfin::uint GenericFunction::value_size() const
{
  uint size = 1;
  for (uint i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
void GenericFunction::evaluate(double* values,
                               const double* coordinates,
                               const ufc::cell& cell) const
{
  assert(values);
  assert(coordinates);

  // Wrap data
  Array<double> _values(value_size(), values);
  const Array<double> x(cell.geometric_dimension, const_cast<double*>(coordinates));

  // Redirect to eval
  eval(_values, x, cell);
}
//-----------------------------------------------------------------------------
void GenericFunction::restrict_as_ufc_function(double* w,
                                               const FiniteElement& element,
                                               const Cell& dolfin_cell,
                                               const ufc::cell& ufc_cell) const
{
  assert(w);

  // Evaluate dofs to get the expansion coefficients
  element.evaluate_dofs(w, *this, ufc_cell);
}
//-----------------------------------------------------------------------------
