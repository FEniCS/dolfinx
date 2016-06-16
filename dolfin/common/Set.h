// Copyright (C) 2009-2011 Garth N. Wells
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
// First added:  2009-08-09
// Last changed: 2011-01-05

#ifndef __DOLFIN_SET_H
#define __DOLFIN_SET_H

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dolfin
{

  /// This is a set-like data structure. It is not ordered and it is based
  /// a std::vector. It uses linear search, and can be faster than std::set
  // and std::unordered_set in some cases.

  template<typename T>
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
    Set(std::vector<T>& x) : _x(x)
    { _x.clear(); }

    /// Copy constructor
    Set(const dolfin::Set<T>& x) : _x(x._x) {}

    /// Destructor
    ~Set() {}

    /// Find entry in set and return an iterator to the entry
    iterator find(const T& x)
    { return std::find(_x.begin(), _x.end(), x); }

    /// Find entry in set and return an iterator to the entry (const)
    const_iterator find(const T& x) const
    { return std::find(_x.begin(), _x.end(), x); }

    /// Insert entry
    bool insert(const T& x)
    {
      if( find(x) == this->end() )
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

    /// Start iterator
    const_iterator begin() const
    { return _x.begin(); }

    /// Iterator beyond end of range
    const_iterator end() const
    { return _x.end(); }

    /// Set size
    std::size_t size() const
    { return _x.size(); }

    /// Erase an entry
    void erase(const T& x)
    {
      iterator p = find(x);
      if (p != _x.end())
        _x.erase(p);
    }

    /// Sort set
    void sort()
    { std::sort(_x.begin(), _x.end()); }

    /// Clear set
    void clear()
    { _x.clear(); }

    /// Index the nth entry in the set
    T operator[](std::size_t n) const
    { return _x[n]; }

    /// Return the vector that stores the data in the Set
    const std::vector<T>& set() const
    { return _x; }

    /// Return the vector that stores the data in the Set
    std::vector<T>& set()
    { return _x; }

  private:

    std::vector<T> _x;

  };

}

#endif
