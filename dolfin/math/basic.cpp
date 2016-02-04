// Copyright (C) 2003-2011 Anders Logg
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
// Modified by Garth N. Wells 2005-2011.
//
// First added:  2003-02-06
// Last changed: 2011-07-01

#include <time.h>
#include <cstdlib>
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include "basic.h"

using namespace dolfin;

namespace dolfin
{
  // Seed only first time
  bool rand_seeded = false;
}

//-----------------------------------------------------------------------------
std::size_t dolfin::ipow(std::size_t a, std::size_t n)
{
  // Treat special case a==0
  if (a == 0)
  {
    if (n == 0)
    {
      // size_t does not have NaN, raise error
      dolfin_error("math/basic.cpp",
                   "take power ipow(0, 0)",
                   "ipow(0, 0) does not have a sense");
    }
    return 0;
  }

  std::size_t p = 1;
  // NOTE: Overflow not checked! Use __builtin_mul_overflow when
  //       GCC >= 5, Clang >= 3.8 is required!
  for (std::size_t i = 0; i < n; i++)
    p *= a;
  return p;
}
//-----------------------------------------------------------------------------
double dolfin::rand()
{
  if (!rand_seeded)
  {
    unsigned int s = static_cast<long int>(time(0));
    std::srand(s);
    rand_seeded = true;
  }
  return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}
//-----------------------------------------------------------------------------
void dolfin::seed(std::size_t s)
{
  std::srand(s);
  rand_seeded = true;
}
//-----------------------------------------------------------------------------
bool dolfin::near(double x, double x0, double eps)
{
  return x0 - eps <= x && x <= x0 + eps;
}
//-----------------------------------------------------------------------------
bool dolfin::between(double x, std::pair<double, double> range)
{
  return range.first - DOLFIN_EPS <= x && x <= range.second + DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
