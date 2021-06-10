// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/IndexMap.h>
#include <memory>

namespace dolfinx::la
{

/// Distributed vector

template <typename T>
class Vector
{
public:
  /// Create vector
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

  /// Scatter local data to ghost values.
  void scatter_fwd()
  {
    xtl::span<const T> xlocal(_x.data(), _map->size_local() * _bs);
    xtl::span<T> xremote(_x.data() + _map->size_local() * _bs,
                         _map->num_ghosts() * _bs);
    _map->scatter_fwd(xlocal, xremote, _bs);
  }

  /// Scatter ghost data to owner. This process will result in multiple
  /// incoming values, which can be summed or inserted into the local vector.
  /// @param op IndexMap operation (add or insert)
  void scatter_rev(dolfinx::common::IndexMap::Mode op)
  {
    xtl::span<T> xlocal(_x.data(), _map->size_local() * _bs);
    xtl::span<const T> xremote(_x.data() + _map->size_local() * _bs,
                               _map->num_ghosts() * _bs);
    _map->scatter_rev(xlocal, xremote, _bs, op);
  }

  /// Inner product of the local part of this vector with the local part
  /// of another vector. To get the global inner product, do a global reduce
  /// of the result with MPI_SUM.
  /// @param b Another la::Vector of the same size
  /// @return Inner product of the local part of this vector with b
  T inner_product(const Vector<T>& b)
  {
    const std::int32_t local_size = _bs * _map->size_local();
    if (b._map->size_local() != _map->size_local() or _bs != b._bs)
      throw std::runtime_error("Incompatible vector for inner_product");

    return std::transform_reduce(_x.begin(), _x.begin() + local_size,
                                 b._x.begin(), 0.0);
  }

  /// L2 Norm of distributed vector
  /// Collective MPI operation
  T norm()
  {
    const std::int32_t size_local = _map->size_local();

    double result = std::transform_reduce(_x.data(), _x.data() + size_local,
                                          0.0, std::plus<double>(),
                                          [](T val) { return std::norm(val); });

    double global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                  _map->comm());

    return std::sqrt(global_result);
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

} // namespace dolfinx::la
