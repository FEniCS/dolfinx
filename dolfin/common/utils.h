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
#include <sstream>
#include <string>
#include <vector>
#include <boost/functional/hash.hpp>
#include <dolfin/common/MPI.h>

namespace dolfin
{

  /// Indent string block
  std::string indent(std::string block);

  /// Return string representation of given container of ints, floats,
  /// etc.
  template<typename T>
    std::string container_to_string(const T& x, std::string delimiter,
                                    int precision)
  {
    std::stringstream s;
    s.precision(precision);
    if (!x.empty())
    {
      s << *x.begin();
      for (auto it = x.begin() + 1; it != x.end(); ++it)
        s << delimiter << *it;
    }
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
    std::size_t hash_global(const MPI_Comm mpi_comm, const T& x)
  {
    // Compute local hash
    std::size_t local_hash = hash_local(x);

    // Gather hash keys on root process
    std::vector<std::size_t> all_hashes;
    std::vector<std::size_t> local_hash_tmp(1, local_hash);
    MPI::gather(mpi_comm, local_hash_tmp, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<std::size_t>> hash;
    std::size_t global_hash = hash(all_hashes);

    // Broadcast hash key to all processes
    MPI::broadcast(mpi_comm, global_hash);
    return global_hash;
  }

}

#endif
