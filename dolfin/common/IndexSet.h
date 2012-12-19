// Copyright (C) 2011 Anders Logg
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
// First added:  2011-02-07
// Last changed: 2011-08-28

#ifndef __INDEX_SET_H
#define __INDEX_SET_H

#include <cstddef>
#include <vector>

namespace dolfin
{

  /// This class provides an efficient data structure for index sets.
  /// The cost of checking whether a given index is in the set is O(1)
  /// and very very fast (optimal) at the cost of extra storage.

  class IndexSet
  {
  public:

    /// Create index set of given size
    IndexSet(std::size_t size) : _size(size), _has_index(size),
                                 _positions(size)
    {
      _indices.reserve(size);
      clear();
    }

    /// Destructor
    ~IndexSet() {}

    /// Return true if set is empty
    bool empty() const
    { return _indices.empty(); }

    /// Return size of set
    std::size_t size() const
    { return _indices.size(); }

    /// Check whether index is in set
    bool has_index(std::size_t index) const
    {
      dolfin_assert(index < _size);
      return _has_index[index];
    }

    /// Return position (if any) for given index
    std::size_t find(std::size_t index) const
    {
      dolfin_assert(index < _size);
      if (!_has_index[index])
        dolfin_error("IndexSet.h",
                     "locate position of index",
                     "Index %d is not in index set", index);
      return _positions[index];
    }

    /// Return given index
    std::size_t& operator[] (std::size_t i)
    {
      dolfin_assert(i < _indices.size());
      return _indices[i];
    }

    /// Return given index (const version)
    const std::size_t& operator[] (std::size_t i) const
    {
      dolfin_assert(i < _indices.size());
      return _indices[i];
    }

    /// Insert index into set
    void insert(std::size_t index)
    {
      dolfin_assert(index < _size);
      if (_has_index[index])
        return;
      _indices.push_back(index);
      _has_index[index] = true;
      _positions[index] = _indices.size() - 1;
    }

    /// Fill index set with indices 0, 1, 2, ..., size - 1
    void fill()
    {
      _indices.clear();
      for (std::size_t i = 0; i < _size; i++)
        _indices.push_back(i);
      std::fill(_has_index.begin(), _has_index.end(), true);
    }

    /// Clear set
    void clear()
    {
      _indices.clear();
      std::fill(_has_index.begin(), _has_index.end(), false);
      std::fill(_positions.begin(), _positions.end(), 0);
    }

  private:

    // Size (maximum index + 1)
    std::size_t _size;

    // Vector of indices
    std::vector<std::size_t> _indices;

    // Indicators for which indices are in the set
    std::vector<std::size_t> _has_index;

    // Mapping from indices to positions
    std::vector<std::size_t> _positions;

  };

}

#endif
