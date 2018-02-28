// Copyright (C) 2013 Johan Hake
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionAXPY.h"
#include "Function.h"
#include "FunctionSpace.h"
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(std::shared_ptr<const Function> func, double scalar)
    : _pairs()
{
  dolfin_assert(func);
  _pairs.push_back(std::make_pair(scalar, func));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy, double scalar) : _pairs()
{
  _register(axpy, scalar);
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(std::shared_ptr<const Function> func0,
                           std::shared_ptr<const Function> func1,
                           Direction direction)
    : _pairs()
{
  dolfin_assert(func0);
  dolfin_assert(func1);
  dolfin_assert(func1->function_space());

  if (*func0->function_space() != *func1->function_space())
  {
    dolfin_error("FunctionAXPY.cpp", "Construct FunctionAXPY",
                 "Expected Functions to be in the same FunctionSpace");
  }

  const double scale0 = static_cast<int>(direction) % 2 == 0 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale0, func0));

  const double scale1 = static_cast<int>(direction) < 2 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale1, func1));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy,
                           std::shared_ptr<const Function> func,
                           Direction direction)
    : _pairs()
{
  dolfin_assert(func);
  dolfin_assert(func->function_space());

  _register(axpy, static_cast<int>(direction) % 2 == 0 ? 1.0 : -1.0);
  if (_pairs.size() > 0
      and *_pairs[0].second->function_space() != *func->function_space())
  {
    dolfin_error("FunctionAXPY.cpp", "Construct FunctionAXPY",
                 "Expected Functions to have the same FunctionSpace");
  }

  const double scale = static_cast<int>(direction) < 2 ? 1.0 : -1.0;
  _pairs.push_back(std::make_pair(scale, func));
}
//-----------------------------------------------------------------------------
FunctionAXPY::FunctionAXPY(const FunctionAXPY& axpy0, const FunctionAXPY& axpy1,
                           Direction direction)
    : _pairs()
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
FunctionAXPY::FunctionAXPY(
    std::vector<std::pair<double, std::shared_ptr<const Function>>> pairs)
    : _pairs(pairs)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionAXPY::~FunctionAXPY()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator+(std::shared_ptr<const Function> func) const
{
  return FunctionAXPY(*this, func, Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator+(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(*this, axpy, Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator-(std::shared_ptr<const Function> func) const
{
  return FunctionAXPY(*this, func, Direction::ADD_SUB);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator-(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(*this, axpy, Direction::ADD_SUB);
}
//-----------------------------------------------------------------------------
const std::vector<std::pair<double, std::shared_ptr<const function::Function>>>&
FunctionAXPY::pairs() const
{
  return _pairs;
}
//-----------------------------------------------------------------------------
void FunctionAXPY::_register(const FunctionAXPY& axpy, double scale)
{
  if (_pairs.size() > 0 and axpy._pairs.size() > 0 and _pairs[0].second
      and axpy._pairs[0].second                   // nullptr checks
      and axpy._pairs[0].second->function_space() // nullptr checks
      and *_pairs[0].second->function_space()
              != *axpy._pairs[0].second->function_space())
  {
    dolfin_error("FunctionAXPY.cpp", "Construct FunctionAXPY",
                 "Expected Functions to have the same FunctionSpace");
  }

  for (auto it = axpy.pairs().begin(); it != axpy.pairs().end(); it++)
  {
    dolfin_assert(it->second);
    _pairs.push_back(std::make_pair(it->first * scale, it->second));
  }
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator*(double scale) const
{
  return FunctionAXPY(*this, scale);
}
//-----------------------------------------------------------------------------
FunctionAXPY FunctionAXPY::operator/(double scale) const
{
  return FunctionAXPY(*this, 1.0 / scale);
}
//-----------------------------------------------------------------------------
