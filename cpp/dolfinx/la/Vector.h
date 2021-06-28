// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <complex>
#include <dolfinx/common/IndexMap.h>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{

/// Distributed vector

template <typename T, class Allocator = std::allocator<T>>
class Vector
{
public:
  /// Create a distributed vector
  Vector(const std::shared_ptr<const common::IndexMap>& map, int bs,
         const Allocator& alloc = Allocator())
      : _map(map), _bs(bs),
        _x(bs * (map->size_local() + map->num_ghosts()), alloc)
  {
    if (bs == 1)
      _datatype = MPI::mpi_type<T>();
    else
    {
      MPI_Type_contiguous(bs, dolfinx::MPI::mpi_type<T>(), &_datatype);
      MPI_Type_commit(&_datatype);
    }
  }

  /// Copy constructor
  Vector(const Vector& x) = default;

  /// Move constructor
  Vector(Vector&& x) noexcept = default;

  /// Destructor
  ~Vector()
  {
    if (_datatype and _datatype != MPI::mpi_type<T>())
      MPI_Type_free(&_datatype);
  }

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Begin scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_begin()
  {
    assert(_map);
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span<const T> xlocal(_x.data(), local_size);
    _map->scatter_fwd_begin(xlocal, _datatype, _request, _buffer_send_fwd,
                            _buffer_recv_fwd);
  }

  /// End scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_end()
  {
    assert(_map);
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span xremote(_x.data() + local_size, _map->num_ghosts() * _bs);
    _map->scatter_fwd_end(xremote, _request,
                          xtl::span<const T>(_buffer_recv_fwd));
  }

  /// Scatter local data to ghost positions on other ranks
  /// @note Collective MPI operation
  void scatter_fwd()
  {
    this->scatter_fwd_begin();
    this->scatter_fwd_end();
  }

  /// Scatter ghost data to owner. This process may receive data from
  /// more than one process, and the received data can be summed or
  /// inserted into the local portion of the vector.
  /// @param op IndexMap operation (add or insert)
  /// @note Collective MPI operation
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
  /// @param type Norm type (supported types are \f$L^2\f$ and \f$L^\infty\f$)
  T norm(la::Norm type = la::Norm::l2) const
  {
    switch (type)
    {
    case la::Norm::l2:
      return std::sqrt(this->squared_norm());
    case la::Norm::linf:
    {
      const std::int32_t size_local = _map->size_local();
      double local_linf = 0.0;
      if (size_local > 0)
      {
        auto local_max_entry = std::max_element(
            _x.begin(), std::next(_x.begin(), size_local),
            [](T a, T b) { return std::norm(a) < std::norm(b); });
        local_linf = std::abs(*local_max_entry);
      }

      double linf = 0.0;
      MPI_Allreduce(&local_linf, &linf, 1, MPI_DOUBLE, MPI_MAX, _map->comm());
      return linf;
    }
    default:
      throw std::runtime_error("Norm type not supported");
    }
  }

  /// Compute the squared L2 norm of vector
  /// @note Collective MPI operation
  double squared_norm() const
  {
    const std::int32_t size_local = _map->size_local();
    double result = std::transform_reduce(
        _x.begin(), std::next(_x.begin(), size_local), 0.0, std::plus<double>(),
        [](T val) { return std::norm(val); });
    double norm2;
    MPI_Allreduce(&result, &norm2, 1, MPI_DOUBLE, MPI_SUM, _map->comm());
    return norm2;
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// Get local part of the vector (const version)
  const std::vector<T, Allocator>& array() const { return _x; }

  /// Get local part of the vector
  std::vector<T, Allocator>& mutable_array() { return _x; }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  // Data type and buffers for ghost scatters
  MPI_Datatype _datatype = MPI_DATATYPE_NULL;
  MPI_Request _request = MPI_REQUEST_NULL;
  std::vector<T> _buffer_send_fwd, _buffer_recv_fwd;

  // Data
  std::vector<T, Allocator> _x;
};

/// Compute the inner product of two vectors. The two vectors must have
/// the same parallel layout
/// @note Collective
/// @param a A vector
/// @param b A vector
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename T, class Allocator = std::allocator<T>>
T inner_product(const Vector<T, Allocator>& a, const Vector<T, Allocator>& b)
{
  const std::int32_t local_size = a.bs() * a.map()->size_local();
  if (local_size != b.bs() * b.map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");
  const std::vector<T>& x_a = a.array();
  const std::vector<T>& x_b = b.array();

  const T local = std::transform_reduce(
      x_a.begin(), x_a.begin() + local_size, x_b.begin(), static_cast<T>(0),
      std::plus<T>(),
      [](T a, T b) -> T
      {
        if constexpr (std::is_same<T, std::complex<double>>::value
                      or std::is_same<T, std::complex<float>>::value)
          return std::conj(a) * b;
        else
          return a * b;
      });

  T result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM,
                a.map()->comm());
  return result;
}

} // namespace dolfinx::la
