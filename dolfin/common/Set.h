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

    /// Copy constructor
    Set(const dolfin::Set<T>& x) : _x(x._x) {}

    iterator find(const T& x)
    { return std::find(_x.begin(), _x.end(), x); }

    void insert(const T& x)
    {
      if( find(x) == this->end() )
        _x.push_back(x);      
    }

    const_iterator begin() const
    { return _x.begin(); }

    const_iterator end() const
    { return _x.end(); }

    const dolfin::uint size() const
    { return _x.size(); }

    void erase(const T& x)
    { 
      if (find(x) != _x.end())
        _x.erase(find(x)); 
    }

    void sort()
    { std::sort(_x.begin(), _x.end()); }

    void clear()
    { _x.clear(); }

    void resize(uint n)
    { _x.resize(n); }

    T operator[](uint n) const
    { return _x[n]; }

    const std::vector<T> set() const
    { return _x; }

  private:

    std::vector<T> _x;

  };

}

#endif
