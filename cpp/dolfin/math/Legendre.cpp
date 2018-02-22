// Copyright (C) 2003-2008 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Legendre.h"
#include <boost/math/special_functions/legendre.hpp>
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
double Legendre::eval(std::size_t n, double x)
{
  return boost::math::legendre_p(n, x);
}
//-----------------------------------------------------------------------------
double Legendre::ddx(std::size_t n, double x)
{
  dolfin_assert(1.0 - x * x > 0.0);
  return -boost::math::legendre_p(n, 1, x) / (std::sqrt(1.0 - x * x));
}
//-----------------------------------------------------------------------------
double Legendre::d2dx(std::size_t n, double x)
{
  dolfin_assert(1.0 - x * x != 0.0);
  return boost::math::legendre_p(n, 2, x) / (1.0 - x * x);
}
//-----------------------------------------------------------------------------
