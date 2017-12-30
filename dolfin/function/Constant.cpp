// Copyright (C) 2006-2011 Anders Logg
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

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include "Constant.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Constant::Constant(double value) : Expression({}), _values(1, value)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(std::vector<double> values)
  : Expression({values.size()}), _values(values)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(std::vector<std::size_t> value_shape,
                   std::vector<double> values)
  : Expression(value_shape), _values(values)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Constant::Constant(const Constant& constant)
  : Expression(constant)
{
  *this = constant;
}
//-----------------------------------------------------------------------------
Constant::~Constant()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator= (const Constant& constant)
{
  // Check value shape
  if (constant.value_shape() != value_shape())
  {
    dolfin_error("Constant.cpp",
                 "assign value to constant",
                 "Value shape mismatch");
  }

  // Assign values
  _values = constant._values;

  return *this;
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator= (double constant)
{
  // Check value shape
  if (!value_shape().empty())
  {
    dolfin_error("Constant.cpp",
                 "assign scalar value to constant",
                 "Constant is not a scalar");
  }

  // Assign value
  dolfin_assert(_values.size() == 1);
  _values[0] = constant;

  return *this;
}
//-----------------------------------------------------------------------------
std::vector<double> Constant::values() const
{
  dolfin_assert(!_values.empty());
  return _values;
}
//-----------------------------------------------------------------------------
void Constant::eval(Eigen::Ref<Eigen::VectorXd> values,
                    Eigen::Ref<const Eigen::VectorXd> x) const
{
  // Copy values
  std::copy(_values.begin(), _values.end(), values.data());
}
//-----------------------------------------------------------------------------
std::string Constant::str(bool verbose) const
{
  std::ostringstream oss;
  oss << "<Constant of dimension " << _values.size() << ">";

  if (verbose)
  {
    std::ostringstream ossv;
    if (!_values.empty())
    {
      ossv << std::endl << std::endl;
      if (!value_shape().empty())
      {
        ossv << "Values: ";
        ossv << "(";
        // Avoid a trailing ", "
        std::copy(_values.begin(), _values.end() - 1,
                  std::ostream_iterator<double>(ossv, ", "));
        ossv << _values.back();
        ossv << ")";
      }
      else
      {
        ossv << "Value: ";
        ossv << _values[0];
      }
    }
    oss << indent(ossv.str());
  }

  return oss.str();
}
//-----------------------------------------------------------------------------
