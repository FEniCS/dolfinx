// Copyright (C) 2009-2011 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dolfin
{

namespace common
{

/// This is a set-like data structure. It is not ordered and it is based
/// a std::vector. It uses linear search, and can be faster than std::set
// and std::unordered_set in some cases.

template <typename T>
class Set
{
public:
  /// Iterator
  typedef typename std::vector<T>::iterator iterator;
  /// Const iterator
  typedef typename std::vector<T>::const_iterator const_iterator;

  /// Create empty set
  Set() {}

  /// Wrap std::vector as a set. Contents will be erased.
  Set(std::vector<T>& x) : _x(x) { _x.clear(); }

  /// Copy constructor
  Set(const dolfin::common::Set<T>& x) : _x(x._x) {}

  /// Destructor
  ~Set() {}

  /// Find entry in set and return an iterator to the entry
  iterator find(const T& x) { return std::find(_x.begin(), _x.end(), x); }

  /// Find entry in set and return an iterator to the entry (const)
  const_iterator find(const T& x) const
  {
    return std::find(_x.begin(), _x.end(), x);
  }

  /// Insert entry
  bool insert(const T& x)
  {
    if (find(x) == this->end())
    {
      _x.push_back(x);
      return true;
    }
    else
      return false;
  }

  /// Insert entries
  template <typename InputIt>
  void insert(const InputIt first, const InputIt last)
  {
    for (InputIt position = first; position != last; ++position)
    {
      if (std::find(_x.begin(), _x.end(), *position) == _x.end())
        _x.push_back(*position);
    }
  }

  /// Iterator to start of Set
  const_iterator begin() const { return _x.begin(); }

  /// Iterator to beyond end of Set
  const_iterator end() const { return _x.end(); }

  /// Set size
  std::size_t size() const { return _x.size(); }

  /// Erase an entry
  void erase(const T& x)
  {
    iterator p = find(x);
    if (p != _x.end())
      _x.erase(p);
  }

  /// Sort set
  void sort() { std::sort(_x.begin(), _x.end()); }

  /// Clear set
  void clear() { _x.clear(); }

  /// Index the nth entry in the set
  T operator[](std::size_t n) const { return _x[n]; }

  /// Return the vector that stores the data in the Set
  const std::vector<T>& set() const { return _x; }

  /// Return the vector that stores the data in the Set
  std::vector<T>& set() { return _x; }

private:
  std::vector<T> _x;
};
}
}