// Copyright (C) 2009-2010 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-08-09
// Last changed: 2010-11-18

#ifndef __UTILS_H
#define __UTILS_H

#include <string>
#include "types.h"

namespace dolfin
{

  /// Indent string block
  std::string indent(std::string block);

  /// Return string representation of int
  std::string to_string(int n);

  /// Return string representation of given array
  std::string to_string(const double* x, uint n);

  /// Return simple hash for given signature string
  dolfin::uint hash(std::string signature);

}

#endif
