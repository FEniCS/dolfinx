// Copyright (C) 2009-2010 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cstring>
#include <dolfinx/common/MPI.h>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dolfinx::common
{

/// Sort two arrays based on the values in array @p indices. Any
/// duplicate indices and the corresponding value are removed. In the
/// case of duplicates, the entry with the smallest value is retained.
/// @param[in] indices Array of indices
/// @param[in] values Array of values
/// @return Sorted (indices, values), with sorting based on indices
template <typename U, typename V>
std::pair<U, V> sort_unique(const U& indices, const V& values)
{
  if (indices.size() != values.size())
    throw std::runtime_error("Cannot sort two arrays of different lengths");

  std::vector<std::pair<typename U::value_type, typename V::value_type>> data(
      indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
    data[i] = {indices[i], values[i]};

  // Sort make unique
  std::sort(data.begin(), data.end());
  auto it = std::unique(data.begin(), data.end(),
                        [](auto& a, auto& b) { return a.first == b.first; });

  U indices_new;
  V values_new;
  indices_new.reserve(data.size());
  values_new.reserve(data.size());
  for (auto d = data.begin(); d != it; ++d)
  {
    indices_new.push_back(d->first);
    values_new.push_back(d->second);
  }

  return {std::move(indices_new), std::move(values_new)};
}

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
} // namespace dolfinx::common
