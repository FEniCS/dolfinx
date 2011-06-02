// Copyright (C) 2004-2005 Anders Logg
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
// First added:  2004-12-21
// Last changed: 2009-08-10

#ifndef __ALLOC_H
#define __ALLOC_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

namespace dolfin
{

  /// This is a special class responsible of allocating data for time
  /// slabs. To get optimal performance with minimal memory usage, all
  /// time slab data structures are simple arrays.
  ///
  /// FIXME: Maybe this should be a template?

  class Alloc
  {
  public:

    /// Constructor
    Alloc();

    /// (Re-)allocate an array of ints
    static void realloc(int** data, uint oldsize, uint newsize);

    /// (Re-)allocate an array of uints
    static void realloc(uint** data, uint oldsize, uint newsize);

    /// (Re-)allocate an array of reals
    static void realloc(real** data, uint oldsize, uint newsize);

    /// Display array of ints
    static void display(uint* data, uint size);

    /// Display array of uints
    static void display(int* data, uint size);

    /// Display array of reals
    static void display(real* data, uint size);

    uint size; // Allocated size
    uint next; // Next available position (used size)

  };

}

#endif
