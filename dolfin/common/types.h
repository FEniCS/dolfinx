// Copyright (C) 2008 Anders Logg
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
// Modified by Garth N. Wells, 2012
//
// First added:  2008-04-22
// Last changed: 2012-11-15
//
// This file provides DOLFIN typedefs for basic types.

#ifndef __DOLFIN_TYPES_H
#define __DOLFIN_TYPES_H

#include <complex>

namespace dolfin
{

  /// Index type for compatibility with linear algebra backend
  typedef int DolfinIndex;

  /// Unsigned integers
  typedef unsigned int uint;

  /// Complex numbers
  typedef std::complex<double> complex;

}

#endif
