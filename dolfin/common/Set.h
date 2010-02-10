// Copyright (C) 2009 Garth N. Wellls.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-09
// Last changed:

#ifndef __DOLFIN_SET_H
#define __DOLFIN_SET_H

#include <algorithm>
#include <vector>
#include "dolfin/common/types.h"

namespace dolfin
{

  /// This is a std::set like data structure. It is not ordered and it is based 
  /// a std::vector. It can be faster than a std::set for some cases.

  template<class T>
  class Set
  {
  public:

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    /// Create empty set
    Set() {}

    /// Wrap std::vectpr as a set. Contents will be erased.
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

    const_iterator begin() const
    { return _x.begin(); }

    const_iterator end() const
    { return _x.end(); }

    /// Set size
    dolfin::uint size() const
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

    /// Resize set
    void resize(uint n)
    { _x.resize(n); }

    /// Index the nth entry in the set
    T operator[](uint n) const
    { return _x[n]; }

    /// Return the vector that stores the data in the Set 
    const std::vector<T> set() const
    { return _x; }

  private:

    std::vector<T> _x;

  };

}

#endif
