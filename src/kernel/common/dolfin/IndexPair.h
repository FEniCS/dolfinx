// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __INDEX_PAIR_H
#define __INDEX_PAIR_H

namespace dolfin
{

  /// IndexPair represents a two-dimensional index pair (i,j)

  class IndexPair
  {
  public:

    /// Create default index pair (0,0)
    IndexPair();

    /// Create given index pair (i,j)
    IndexPair(unsigned int i, unsigned int j);

    /// Destructor
    ~IndexPair();

    /// The pair of indices
    unsigned int i, j;
    
  };

}

#endif
