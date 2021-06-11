// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <memory>

namespace dolfinx::la
{

/// Distributed vector

template <typename T>
class Vector
{
public:
  /// Create a distributed vector
  Vector(const std::shared_ptr<const common::IndexMap>& map, int bs)
      : _map(map), _bs(bs)
  {
    assert(map);
    const std::int32_t local_size
        = bs * (map->size_local() + map->num_ghosts());
    _x.resize(local_size);
  }

  /// Copy constructor
  Vector(const Vector& x) = default;

  /// Move constructor
  Vector(Vector&& x) noexcept = default;

  /// Destructor
  ~Vector() = default;

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Scatter local data to ghost values
  void scatter_fwd()
  {
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span<const T> xlocal(_x.data(), local_size);
    xtl::span xremote(_x.data() + local_size, _map->num_ghosts() * _bs);
    _map->scatter_fwd(xlocal, xremote, _bs);
  }

  /// Scatter ghost data to owner. This process will result in multiple
  /// incoming values, which can be summed or inserted into the local
  /// vector.
  /// @param op IndexMap operation (add or insert)
  void scatter_rev(dolfinx::common::IndexMap::Mode op)
  {
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span xlocal(_x.data(), local_size);
    xtl::span<const T> xremote(_x.data() + local_size,
                               _map->num_ghosts() * _bs);
    _map->scatter_rev(xlocal, xremote, _bs, op);
  }

  /// Compute the norm of the vector
  /// @note Collective MPI operation
  T norm(la::Norm type = la::Norm::l2)
  {
    switch (type)
    {
    case la::Norm::l2:
      return std::sqrt(this->squared_norm);
    default:
      throw std::runtime_error("Norm type not supported");
    }
  }

  /// Compute the squared L2 norm of vector
  /// @note Collective MPI operation
  T squared_norm()
  {
    const std::int32_t size_local = _map->size_local();
    double result = std::transform_reduce(_x.data(), _x.data() + size_local,
                                          0.0, std::plus<double>(),
                                          [](T val) { return std::norm(val); });
    double norm2;
    MPI_Allreduce(&result, &norm2, 1, MPI_DOUBLE, MPI_SUM, _map->comm());
    return norm2;
  }

  /// Maximum value of the local part of the vector. To get the global maximum
  /// do a global reduction with MPI_MAX
  T max()
  {
    static_assert(!std::is_same<T, std::complex<double>>::value
                      and !std::is_same<T, std::complex<float>>::value,
                  "max cannot be used with complex.");
    const std::int32_t size_local = _map->size_local();
    T result = std::reduce(_x.data(), _x.data() + size_local, 0.0,
                           [](T a, T b) { return std::max(a, b); });
    return result;
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// Get local part of the vector (const version)
  const std::vector<T>& array() const { return _x; }

  /// Get local part of the vector
  std::vector<T>& mutable_array() { return _x; }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  // Data
  std::vector<T> _x;
};

/// Compute the inner product of two vectors. The two vectors must have
/// the same parallel layout
/// @note Collective
/// @param a A vector
/// @param b A vector
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename T>
T inner_product(const Vector<T>& a, const Vector<T>& b)
{
  const std::int32_t local_size = a.bs() * a.map()->size_local();
  if (local_size != b.bs() * b.map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");
  const std::vector<T>& x_a = a.array();
  const std::vector<T>& x_b = b.array();
  T local
      = std::transform_reduce(a.begin(), a.begin() + local_size, b.begin(), 0.0,
                              [](T a, T b) { return std::conj(a) * b; });
  double result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM,
                a->map().comm());
  return result;
}

} // namespace dolfinx::la
