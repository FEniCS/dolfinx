// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __INDEX_H
#define __INDEX_H

namespace dolfin
{

  /// IndexPair represents a two-dimensional index pair (i,j)

  class Index
  {
  public:

    /// Create default index (0,0)
    Index();

    /// Create given index
    Index(unsigned int i, unsigned int j);

    /// Destructor
    ~Index();

    /// Index pair (i,j)
    unsigned int i, j;
    
  };

}

#endif
