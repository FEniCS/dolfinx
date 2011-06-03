// Copyright (C) 2003-2005 Anders Logg
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
// First added:  2003-02-06
// Last changed: 2006-06-16

#ifndef __BASIC_H
#define __BASIC_H

#include <cmath>
#include <dolfin/common/types.h>

namespace dolfin
{
  /// Return a to the power n
  uint ipow(uint a, uint n);

  /// Return a random number, uniformly distributed between [0.0, 1.0)
  double rand();

  /// Seed random number generator
  void seed(unsigned int s);

}

#endif
