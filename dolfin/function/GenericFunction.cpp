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
void GenericFunction::eval(Eigen::Ref<Eigen::VectorXd> values,
                           Eigen::Ref<const Eigen::VectorXd> x,
                           const ufc::cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void GenericFunction::eval(Eigen::Ref<Eigen::VectorXd> values,
                           Eigen::Ref<const Eigen::VectorXd> x) const
{
  dolfin_error("GenericFunction.cpp",
               "evaluate function (Eigen version)",
               "Missing eval() function (must be overloaded)");
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
  Eigen::Map<Eigen::VectorXd> _values(values, value_size());
  Eigen::Map<const Eigen::VectorXd> x(coordinates, cell.geometric_dimension);

  // Redirect to eval
  eval(_values, x, cell);
}
//-----------------------------------------------------------------------------
