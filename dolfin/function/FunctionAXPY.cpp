// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-11
// Last changed: 2013-02-15

#include <dolfin/log/log.h>
#include "Function.h"
#include "FunctionSpace.h"
#include "FunctionAXPY.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const Function& func, double scalar) : _pairs()
{
  _pairs.push_back(std::make_pair(scalar, &func));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy, double scalar) : _pairs()
{
  _register(axpy, scalar);
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const Function& func0, const Function& func1,
			   Direction direction) : _pairs()
{
  if (!func0.in(*func1.function_space()))
  {
    dolfin_error("FunctionAXPY.cpp",
                 "Construct FunctionAXPY",
                 "Expected Functions to be in the same FunctionSpace");
  }

  const double scale0 = static_cast<int>(direction) % 2 == 0 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale0, &func0));

  const double scale1 = static_cast<int>(direction) < 2 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale1, &func1));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy, const Function& func,
			   Direction direction) : _pairs()
{
  _register(axpy, static_cast<int>(direction) % 2 == 0 ? 1.0 : -1.0);
  if (_pairs.size()>0 && !_pairs[0].second->in(*func.function_space()))
  {
    dolfin_error("FunctionAXPY.cpp",
                 "Construct FunctionAXPY",
                 "Expected Functions to have the same FunctionSpace");
  }

  const double scale = static_cast<int>(direction) < 2 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale, &func));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy0,
                           const FunctionAXPY& axpy1,
			   Direction direction) : _pairs()
{
  _register(axpy0, static_cast<int>(direction) % 2 == 0 ? 1.0 : -1.0);
  _register(axpy1, static_cast<int>(direction) < 2 ? 1.0 : -1.0);
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy) : _pairs(axpy._pairs)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(std::vector<std::pair<double,
                           const Function*>> pairs) : _pairs(pairs)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionAXPY::~FunctionAXPY()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator+(const Function& func) const
{
  return FunctionAXPY(*this, func, Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator+(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(*this, axpy, Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator-(const Function& func) const
{
  return FunctionAXPY(*this, func, Direction::ADD_SUB);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator-(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(*this, axpy, Direction::ADD_SUB);
}
//-----------------------------------------------------------------------------
const std::vector<std::pair<double, const Function*>>&
  FunctionAXPY::pairs() const
{
  return _pairs;
}
//-----------------------------------------------------------------------------
void FunctionAXPY::_register(const FunctionAXPY& axpy, double scale)
{
  if (_pairs.size() > 0 && axpy._pairs.size() > 0
      &&!_pairs[0].second->in(*axpy._pairs[0].second->function_space()))
  {
    dolfin_error("FunctionAXPY.cpp",
                 "Construct FunctionAXPY",
                 "Expected Functions to have the same FunctionSpace");
  }

  for (auto it = axpy.pairs().begin(); it != axpy.pairs().end(); it++)
    _pairs.push_back(std::make_pair(it->first*scale, it->second));
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator*(double scale) const
{
  return FunctionAXPY(*this, scale);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator/(double scale) const
{
  return FunctionAXPY(*this, 1.0/scale);
}
//-----------------------------------------------------------------------------
