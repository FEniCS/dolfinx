// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Constant.h"
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

using namespace dolfin;
using namespace dolfin::function;

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
Constant::Constant(const Constant& constant) : Expression(constant)
{
  *this = constant;
}
//-----------------------------------------------------------------------------
Constant::~Constant()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator=(const Constant& constant)
{
  // Check value shape
  if (constant.value_shape() != value_shape())
  {
    log::dolfin_error("Constant.cpp", "assign value to constant",
                      "Value shape mismatch");
  }

  // Assign values
  _values = constant._values;

  return *this;
}
//-----------------------------------------------------------------------------
const Constant& Constant::operator=(double constant)
{
  // Check value shape
  if (!value_shape().empty())
  {
    log::dolfin_error("Constant.cpp", "assign scalar value to constant",
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
void Constant::eval(Eigen::Ref<EigenRowArrayXXd> values,
                    Eigen::Ref<const EigenRowArrayXXd> x) const
{
  // Copy values
  for (unsigned int i = 0; i != values.rows(); ++i)
    std::copy(_values.begin(), _values.end(),
              values.data() + i * _values.size());
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
    oss << common::indent(ossv.str());
  }

  return oss.str();
}
//-----------------------------------------------------------------------------
