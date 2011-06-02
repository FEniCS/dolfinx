// Copyright (C) 2009 Shawn W. Walker
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
// This file is to be used as a "testing-ground" for future additions to
// ufc.h (see the UFC project for more info).  That means when using the
// functionality in here, one must use dynamic_cast to access the
// data structures created here.
//
// First added:  2009-04-30
// Last changed: 2009-04-30

#ifndef __UFCEXP_H
#define __UFCEXP_H

#include <ufc.h>

namespace ufcexp
{

  /// This class defines the data structure for a cell in a mesh.

  class cell : public ufc::cell
  {
  public:

    /// Constructor
    cell(): ufc::cell(), higher_order_coordinates(0) {}

    /// Destructor
    virtual ~cell() {}

    /// Array of coordinates for higher order vertices of the cell
    double** higher_order_coordinates;

  };

}

#endif
