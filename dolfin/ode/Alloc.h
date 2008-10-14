// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-12-21
// Last changed: 2005

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
    static void disp(uint* data, uint size);

    /// Display array of uints
    static void disp(int* data, uint size);

    /// Display array of reals
    static void disp(real* data, uint size);

    uint size; // Allocated size
    uint next; // Next available position (used size)
  };
  
}

#endif
