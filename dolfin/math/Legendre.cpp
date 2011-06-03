// Copyright (C) 2003-2008 Anders Logg
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
// Modified by Benjamin Kehlet
//
// First added:  2003-06-03
// Last changed: 2009-02-17

#include <cmath>
#include <boost/math/special_functions/legendre.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/real.h>
#include <dolfin/log/dolfin_log.h>
#include "Legendre.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
real Legendre::eval(uint n, real x)
{
  return boost::math::legendre_p(n, x);
}
//-----------------------------------------------------------------------------
real Legendre::ddx(uint n, real x)
{
  assert(1.0 - x*x > 0.0);
  return -boost::math::legendre_p(n, 1, x)/(std::sqrt(1.0 - x*x));
}
//-----------------------------------------------------------------------------
real Legendre::d2dx(uint n, real x)
{
  assert(1.0 - x*x != 0.0);
  return boost::math::legendre_p(n, 2, x)/(1.0 - x*x);
}
//-----------------------------------------------------------------------------
