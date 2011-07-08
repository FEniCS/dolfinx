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
#include "basic.h"

using namespace dolfin;

namespace dolfin
{
  // Seed only first time
  bool rand_seeded = false;
}

//-----------------------------------------------------------------------------
dolfin::uint dolfin::ipow(uint a, uint n)
{
  uint p = a;
  for (uint i = 1; i < n; i++)
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
void dolfin::seed(unsigned int s)
{
  std::srand(s);
  rand_seeded = true;
}
//-----------------------------------------------------------------------------
bool dolfin::near(double x, double x0, double eps)
{
  return x0 - eps < x && x < x0 + eps;
  //return x0 - eps <= x && x <= x0 + eps;
}
//-----------------------------------------------------------------------------
//bool between(double x, std::pair<double, double> range)
bool dolfin::between(double x0, double x, double x1, double eps)
{
  return x0 - eps <= x && x <= x1 + eps;
  //return range.first <= x && x <= range.second;
}
//-----------------------------------------------------------------------------
