// Copyright (C) 2014 Garth N. Wells
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

#ifndef __TEST_NULLSPACE_H
#define __TEST_NULLSPACE_H

#include <string>

namespace dolfin
{

  // Forward declarations
  class GenericLinearOperator;
  class VectorSpaceBasis;

  /// Check whether a vector space basis is in the nullspace of a
  /// given operator. The string option 'type' can be "right" for the
  /// right nullspace (Ax=0) or "left" for the left nullspace (A^Tx =
  /// 0). To test the left nullspace, A must also be of type
  /// GenericMatrix.
  bool in_nullspace(const GenericLinearOperator& A, const VectorSpaceBasis& x,
                    std::string type="right");
}

#endif
