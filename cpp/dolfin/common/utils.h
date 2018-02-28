// Copyright (C) 2009-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/functional/hash.hpp>
#include <cstring>
#include <dolfin/common/MPI.h>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dolfin
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
      for (unsigned int i = 0; i != x.size(); ++i)
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
}