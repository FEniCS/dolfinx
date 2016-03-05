// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-11-09
// Last changed:

#ifndef __GENERIC_PRECONDITIONER_H
#define __GENERIC_PRECONDITIONER_H

#include <vector>
#include <dolfin/log/log.h>

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class VectorSpaceBasis;

  /// This class provides a common base preconditioners.

  class GenericPreconditioner
  {
  public:

    virtual ~GenericPreconditioner() {}

    /// Set the coordinates of the operator (matrix) rows and
    /// geometric dimension d. This is can be used by required for
    /// certain preconditioners, e.g. ML. The input for this function
    /// can be generated using GenericDofMap::tabulate_all_dofs.
    virtual void set_coordinates(const std::vector<double>& x, std::size_t dim)
    {
      dolfin_error("GenericPreconditioner.h",
                   "set coordinates for preconditioner operator",
                   "Not supported by current preconditioner type");
    }

  };
}

#endif
