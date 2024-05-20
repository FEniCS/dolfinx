// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <cmath>
#include <complex>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/types.h>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace dolfinx::la
{

/// Distributed vector
///
/// @tparam T Scalar type
/// @tparam Container data container type
template <typename T, typename Container = std::vector<T>>
class Vector
{
  static_assert(std::is_same_v<typename Container::value_type, T>);

public:
  /// Scalar type
  using value_type = T;

  /// Container type
  using container_type = Container;

  static_assert(std::is_same_v<value_type, typename container_type::value_type>,
                "Scalar type and container value type must be the same.");

  /// Create a distributed vector
  /// @param map IndexMap for parallel distribution of the data
  /// @param bs Block size
  Vector(std::shared_ptr<const common::IndexMap> map, int bs)
      : _map(map), _scatterer(std::make_shared<common::Scatterer<>>(*_map, bs)),
        _bs(bs), _buffer_local(_scatterer->local_buffer_size()),
        _buffer_remote(_scatterer->remote_buffer_size()),
        _x(bs * (map->size_local() + map->num_ghosts()))
  {
  }

  /// Copy constructor
  Vector(const Vector& x)
      : _map(x._map), _scatterer(x._scatterer), _bs(x._bs),
        _request(1, MPI_REQUEST_NULL), _buffer_local(x._buffer_local),
        _buffer_remote(x._buffer_remote), _x(x._x)
  {
  }

  /// Move constructor
  Vector(Vector&& x)
      : _map(std::move(x._map)), _scatterer(std::move(x._scatterer)),
        _bs(std::move(x._bs)),
        _request(std::exchange(x._request, {MPI_REQUEST_NULL})),
        _buffer_local(std::move(x._buffer_local)),
        _buffer_remote(std::move(x._buffer_remote)), _x(std::move(x._x))
  {
  }

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Set all entries (including ghosts)
  /// @param[in] v The value to set all entries to (on calling rank)
  void set(value_type v) { std::fill(_x.begin(), _x.end(), v); }

  /// Begin scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_begin()
  {
    const std::int32_t local_size = _bs * _map->size_local();
    std::span<const value_type> x_local(_x.data(), local_size);

    auto pack = [](auto&& in, auto&& idx, auto&& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
    pack(x_local, _scatterer->local_indices(), _buffer_local);

    _scatterer->scatter_fwd_begin(std::span<const value_type>(_buffer_local),
                                  std::span<value_type>(_buffer_remote),
                                  std::span<MPI_Request>(_request));
  }

  /// End scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_end()
  {
    const std::int32_t local_size = _bs * _map->size_local();
    const std::int32_t num_ghosts = _bs * _map->num_ghosts();
    std::span<value_type> x_remote(_x.data() + local_size, num_ghosts);
    _scatterer->scatter_fwd_end(std::span<MPI_Request>(_request));

    auto unpack = [](auto&& in, auto&& idx, auto&& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };

    unpack(_buffer_remote, _scatterer->remote_indices(), x_remote,
           [](auto /*a*/, auto b) { return b; });
  }

  /// Scatter local data to ghost positions on other ranks
  /// @note Collective MPI operation
  void scatter_fwd()
  {
    this->scatter_fwd_begin();
    this->scatter_fwd_end();
  }

  /// Start scatter of  ghost data to owner
  /// @note Collective MPI operation
  void scatter_rev_begin()
  {
    const std::int32_t local_size = _bs * _map->size_local();
    const std::int32_t num_ghosts = _bs * _map->num_ghosts();
    std::span<value_type> x_remote(_x.data() + local_size, num_ghosts);

    auto pack = [](auto&& in, auto&& idx, auto&& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
    pack(x_remote, _scatterer->remote_indices(), _buffer_remote);

    _scatterer->scatter_rev_begin(std::span<const value_type>(_buffer_remote),
                                  std::span<value_type>(_buffer_local),
                                  _request);
  }

  /// End scatter of ghost data to owner. This process may receive data
  /// from more than one process, and the received data can be summed or
  /// inserted into the local portion of the vector.
  /// @param op The operation to perform when adding/setting received
  /// values (add or insert)
  /// @note Collective MPI operation
  template <class BinaryOperation>
  void scatter_rev_end(BinaryOperation op)
  {
    const std::int32_t local_size = _bs * _map->size_local();
    std::span<value_type> x_local(_x.data(), local_size);
    _scatterer->scatter_rev_end(_request);

    auto unpack = [](auto&& in, auto&& idx, auto&& out, auto op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
    unpack(_buffer_local, _scatterer->local_indices(), x_local, op);
  }

  /// Scatter ghost data to owner. This process may receive data from
  /// more than one process, and the received data can be summed or
  /// inserted into the local portion of the vector.
  /// @param op IndexMap operation (add or insert)
  /// @note Collective MPI operation
  template <class BinaryOperation>
  void scatter_rev(BinaryOperation op)
  {
    this->scatter_rev_begin();
    this->scatter_rev_end(op);
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> index_map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// Get local part of the vector (const version)
  std::span<const value_type> array() const
  {
    return std::span<const value_type>(_x);
  }

  /// Get local part of the vector
  std::span<value_type> mutable_array() { return std::span(_x); }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Scatter for managing MPI communication
  std::shared_ptr<const common::Scatterer<>> _scatterer;

  // Block size
  int _bs;

  // MPI request handle
  std::vector<MPI_Request> _request = {MPI_REQUEST_NULL};

  // Buffers for ghost scatters
  container_type _buffer_local, _buffer_remote;

  // Vector data
  container_type _x;
};

/// Compute the inner product of two vectors. The two vectors must have
/// the same parallel layout
/// @note Collective MPI operation
/// @param a A vector
/// @param b A vector
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <class V>
auto inner_product(const V& a, const V& b)
{
  using T = typename V::value_type;
  const std::int32_t local_size = a.bs() * a.index_map()->size_local();
  if (local_size != b.bs() * b.index_map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");
  std::span<const T> x_a = a.array().subspan(0, local_size);
  std::span<const T> x_b = b.array().subspan(0, local_size);

  const T local = std::transform_reduce(
      x_a.begin(), x_a.end(), x_b.begin(), static_cast<T>(0), std::plus{},
      [](T a, T b) -> T
      {
        if constexpr (std::is_same<T, std::complex<double>>::value
                      or std::is_same<T, std::complex<float>>::value)
        {
          return std::conj(a) * b;
        }
        else
          return a * b;
      });

  T result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_type<T>(), MPI_SUM,
                a.index_map()->comm());
  return result;
}

/// Compute the squared L2 norm of vector
/// @note Collective MPI operation
template <class V>
auto squared_norm(const V& a)
{
  using T = typename V::value_type;
  T result = inner_product(a, a);
  return std::real(result);
}

/// Compute the norm of the vector
/// @note Collective MPI operation
/// @param x A vector
/// @param type Norm type
template <class V>
auto norm(const V& x, Norm type = Norm::l2)
{
  using T = typename V::value_type;
  switch (type)
  {
  case Norm::l1:
  {
    std::int32_t size_local = x.bs() * x.index_map()->size_local();
    std::span<const T> data = x.array().subspan(0, size_local);
    using U = typename dolfinx::scalar_value_type_t<T>;
    U local_l1
        = std::accumulate(data.begin(), data.end(), U(0),
                          [](auto norm, auto x) { return norm + std::abs(x); });
    U l1(0);
    MPI_Allreduce(&local_l1, &l1, 1, MPI::mpi_type<U>(), MPI_SUM,
                  x.index_map()->comm());
    return l1;
  }
  case Norm::l2:
    return std::sqrt(squared_norm(x));
  case Norm::linf:
  {
    std::int32_t size_local = x.bs() * x.index_map()->size_local();
    std::span<const T> data = x.array().subspan(0, size_local);
    auto max_pos = std::max_element(data.begin(), data.end(), [](T a, T b)
                                    { return std::norm(a) < std::norm(b); });
    auto local_linf = std::abs(*max_pos);
    decltype(local_linf) linf = 0;
    MPI_Allreduce(&local_linf, &linf, 1, MPI::mpi_type<decltype(linf)>(),
                  MPI_MAX, x.index_map()->comm());
    return linf;
  }
  default:
    throw std::runtime_error("Norm type not supported");
  }
}

/// Orthonormalize a set of vectors
/// @param[in,out] basis The set of vectors to orthonormalise. The
/// vectors must have identical parallel layouts. The vectors are
/// modified in-place.
/// @tparam V dolfinx::la::Vector
template <class V>
void orthonormalize(std::vector<std::reference_wrapper<V>> basis)
{
  using T = typename V::value_type;
  using U = typename dolfinx::scalar_value_type_t<T>;

  // Loop over each vector in basis
  for (std::size_t i = 0; i < basis.size(); ++i)
  {
    // Orthogonalize vector i with respect to previously orthonormalized
    // vectors
    V& bi = basis[i].get();
    for (std::size_t j = 0; j < i; ++j)
    {
      const V& bj = basis[j].get();

      // basis_i <- basis_i - dot_ij  basis_j
      auto dot_ij = inner_product(bi, bj);
      std::transform(bj.array().begin(), bj.array().end(), bi.array().begin(),
                     bi.mutable_array().begin(),
                     [dot_ij](auto xj, auto xi) { return xi - dot_ij * xj; });
    }

    // Normalise basis function
    auto norm = la::norm(bi, la::Norm::l2);
    if (norm * norm < std::numeric_limits<U>::epsilon())
    {
      throw std::runtime_error(
          "Linear dependency detected. Cannot orthogonalize.");
    }
    std::transform(bi.array().begin(), bi.array().end(),
                   bi.mutable_array().begin(),
                   [norm](auto x) { return x / norm; });
  }
}

/// @brief Test if basis is orthonormal.
///
/// Returns true if ||x_i - x_j|| - delta_{ij} < eps fro all i, j, and
/// otherwise false.
///
/// @param[in] basis Set of vectors to check.
/// @param[in] eps Tolerance.
/// @return True is basis is orthonormal, otherwise false.
template <class V>
bool is_orthonormal(
    std::vector<std::reference_wrapper<const V>> basis,
    dolfinx::scalar_value_type_t<typename V::value_type> eps
    = std::numeric_limits<
        dolfinx::scalar_value_type_t<typename V::value_type>>::epsilon())
{
  using T = typename V::value_type;
  for (std::size_t i = 0; i < basis.size(); i++)
  {
    for (std::size_t j = i; j < basis.size(); j++)
    {
      T delta_ij = (i == j) ? T(1) : T(0);
      auto dot_ij = inner_product(basis[i].get(), basis[j].get());
      if (std::norm(delta_ij - dot_ij) > eps)
        return false;
    }
  }

  return true;
}

} // namespace dolfinx::la
