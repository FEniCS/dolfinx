// Copyright (C) 2012 Joachim B Haga
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <dolfin/log/log.h>
#include <vector>

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
  explicit RangedIndexSet(std::array<std::int64_t, 2> range)
      : _range(range), _is_set(range[1] - range[0], false)
  {
  }

  /// Return true if a given index is within range, i.e., if it can
  /// be stored in the set.
  bool in_range(std::int64_t i) const
  {
    return (i >= _range[0] && i < _range[1]);
  }

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
  // void clear()
  //{ std::fill(_is_set.begin(), _is_set.end(), false); }

private:
  const std::array<std::int64_t, 2> _range;
  std::vector<bool> _is_set;
};
}


