// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-07

#ifndef __INDEX_SET_H
#define __INDEX_SET_H

#include <vector>
#include "types.h"

namespace dolfin
{

  /// This class provides an efficient data structure for index sets.
  /// The cost of checking whether a given index is in the set is O(1)
  /// and very very fast (optimal) at the cost of extra storage.

  class IndexSet
  {
  public:

    /// Create index set of given size
    IndexSet(uint size) : _has_index(size)
    {
      _indices.reserve(size);
      clear();
    }

    /// Return size of set
    uint size() const
    { return _indices.size(); }

    /// Check whether index is in set
    bool has_index(uint index) const
    { return _has_index[index]; }

    /// Return given index
    uint& operator[] (uint i)
    { return _indices[i]; }

    /// Return given index (const version)
    const uint& operator[] (uint i) const
    { return _indices[i]; }

    /// Insert index into set
    void insert(uint index)
    {
      if (_has_index[index])
        return;
      _indices.push_back(index);
      _has_index[index] = true;
    }

    /// Clear set
    void clear()
    {
      _indices.clear();
      std::fill(_has_index.begin(), _has_index.end(), false);
    }

  private:

    // Vector of indices
    std::vector<uint> _indices;

    // Indicators for which indices are in the set
    std::vector<uint> _has_index;

  };

}

#endif
