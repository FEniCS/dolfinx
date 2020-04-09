// Copyright (C) 2009-2010 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/functional/hash.hpp>
#include <cstring>
#include <dolfinx/common/MPI.h>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace dolfinx
{
namespace common
{

/// Indent string block
std::string indent(std::string block);

/// Return string representation of given container of ints, floats,
/// etc.
template <typename T>
std::string container_to_string(const T& x, std::string delimiter,
                                int precision, int linebreak = 0)
{
  std::stringstream s;
  s.precision(precision);
  if (!x.empty())
  {
    if (linebreak == 0)
    {
      s << *x.begin();
      for (auto it = x.begin() + 1; it != x.end(); ++it)
        s << delimiter << *it;
    }
    else
    {
      for (std::size_t i = 0; i != x.size(); ++i)
      {
        if ((i + 1) % linebreak == 0)
          s << x[i] << std::endl;
        else
          s << x[i] << delimiter;
      }
    }
  }
  return s.str();
}

/// Return a hash of a given object
template <class T>
std::size_t hash_local(const T& x)
{
  boost::hash<T> hash;
  return hash(x);
}

/// Return a hash for a distributed (MPI) object. A hash is computed on
/// each process, and the hash of the std::vector of all local hash keys
/// is returned. This function is collective.
template <class T>
std::int64_t hash_global(const MPI_Comm comm, const T& x)
{
  // Compute local hash
  int64_t local_hash = hash_local(x);

  // Gather hash keys on root process
  std::vector<int64_t> all_hashes(dolfinx::MPI::size(comm));
  MPI_Gather(&local_hash, 1, MPI_INT64_T, all_hashes.data(), 1, MPI_INT64_T, 0,
             comm);

  // Hash the received hash keys
  boost::hash<std::vector<int64_t>> hash;
  int64_t global_hash = hash(all_hashes);

  // Broadcast hash key to all processes
  MPI_Bcast(&global_hash, 1, MPI_INT64_T, 0, comm);

  return global_hash;
}
} // namespace common
} // namespace dolfinx
