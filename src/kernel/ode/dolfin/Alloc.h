// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ALLOC_H
#define __ALLOC_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// This is a special class responsible of allocating data for time
  /// slabs. To get optimal performance with minimal memory usage, all
  /// time slab data structures are simple arrays.
  
  class Alloc
  {
  public:
    
    /// Constructor
    Alloc();
    
    /// (Re-)allocate an array of reals
    static void realloc(real** data, uint oldsize, uint newsize);
    
    /// (Re-)allocate an array of uints
    static void realloc(uint** data, uint oldsize, uint newsize);

    uint size; // Allocated size
    uint next; // Next available position (used size)
  };
  
}

#endif
