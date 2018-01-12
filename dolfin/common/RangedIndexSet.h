// Copyright (C) 2012 Joachim B Haga
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
// First added:  2012-03-02
// Last changed: 2012-03-02

#ifndef __RANGED_INDEX_SET_H
#define __RANGED_INDEX_SET_H

#include <array>
#include <cstddef>
#include <vector>
#include <dolfin/log/log.h>

namespace dolfin
{

  /// This class provides an special-purpose data structure for
  /// testing if a given index within a range is set.
  ///
  /// The memory requirements are one bit per item in range, since it
  /// uses a (packed) std::vector<bool> for storage.

  class RangedIndexSet
  {

  public:

    /// Create a ranged set with range given as a (lower, upper) pair
    explicit RangedIndexSet(std::array<std::int64_t, 2> range) : _range(range),
      _is_set(range[1] - range[0], false) {}

    /// Return true if a given index is within range, i.e., if it can
    /// be stored in the set.
    bool in_range(std::int64_t i) const
    { return (i >= _range[0] && i < _range[1]); }

    /// Check is the set contains the given index.
    bool has_index(std::int64_t i) const
    {
      dolfin_assert(in_range(i));
      return _is_set[i - _range[0]];
    }

    /// Insert a given index into the set. Returns true if the index
    /// was inserted (i.e., the index was not already in the set).
    bool insert(std::int64_t i)
    {
      dolfin_assert(in_range(i));
      std::vector<bool>::reference entry = _is_set[i - _range[0]];
      if (entry)
        return false;
      else
      {
        entry = true;
        return true;
      }
    }

    /// Erase an index from the set.
    void erase(std::int64_t i)
    {
      dolfin_assert(in_range(i));
      _is_set[i - _range[0]] = false;
    }

    /// Erase all indices from the set.
    //void clear()
    //{ std::fill(_is_set.begin(), _is_set.end(), false); }

  private:

    const std::array<std::int64_t, 2> _range;
    std::vector<bool> _is_set;

  };

}

#endif
