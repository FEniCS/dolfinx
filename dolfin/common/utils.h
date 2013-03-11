// Copyright (C) 2009-2010 Anders Logg
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
// Modified by Joachim B. Haga, 2012.
// Modified by Garth N. Wells, 2013.
//
// First added:  2009-08-09
// Last changed: 2013-01-03

#ifndef __DOLFIN_UTILS_H
#define __DOLFIN_UTILS_H

#include <cstring>
#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include <boost/functional/hash.hpp>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  /// Indent string block
  std::string indent(std::string block);

  /// Return string representation of int
  template <class T>
  std::string to_string(T x)
  {
    std::stringstream s;
    s << x;
    return s.str();
  }

  /// Return string representation of given array
  std::string to_string(const double* x, std::size_t n);

  /// Return a hash of a given object
  template <class T>
  std::size_t hash_local(const T& x)
  {
    boost::hash<T> hash;
    return hash(x);
  }

  /// Return a hash for a distributed (MPI) object. A hash is computed
  /// on each process, and the hash of the std::vector of all local hash
  /// keys is returned. This function is collective.
  template <class T>
  std::size_t hash_global(const T& x)
  {
    // Compute local hash
    std::size_t local_hash = hash_local(x);

    // Gather hash keys on root process
    std::vector<std::size_t> all_hashes;
    MPI::gather(local_hash, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<std::size_t> > hash;
    const std::size_t global_hash = hash(all_hashes);

    // Broadcast hash key to all processes
    MPI::broadcast(global_hash);
    return global_hash;
  }

  /// Fast zero-fill of numeric vectors/blocks.
  template <class T>
  void zerofill(T* arr, std::size_t n)
  {
    if (std::numeric_limits<T>::is_integer || std::numeric_limits<T>::is_iec559)
      std::memset(arr, 0, n*sizeof(T));
    else
      // should never happen in practice
      std::fill_n(arr, n, T(0));
  }

  template <class T>
  void zerofill(std::vector<T>& vec)
  { zerofill(&vec[0], vec.size()); }

}

#endif
