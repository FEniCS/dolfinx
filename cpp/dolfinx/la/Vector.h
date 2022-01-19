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
  /// The value type
  using value_type = T;

  /// The allocator type
  using allocator_type = Allocator;

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
  Vector(const Vector& x)
      : _map(x._map), _bs(x._bs), _request(MPI_REQUEST_NULL),
        _buffer_send_fwd(x._buffer_send_fwd),
        _buffer_recv_fwd(x._buffer_recv_fwd), _x(x._x)
  {
    MPI_Type_dup(x._datatype, &_datatype);
  }

  /// Move constructor
  Vector(Vector&& x)
      : _map(std::move(x._map)), _bs(std::move(x._bs)),
        _datatype(std::exchange(x._datatype, MPI_DATATYPE_NULL)),
        _request(std::exchange(x._request, MPI_REQUEST_NULL)),
        _buffer_send_fwd(std::move(x._buffer_send_fwd)),
        _buffer_recv_fwd(std::move(x._buffer_recv_fwd)), _x(std::move(x._x))
  {
  }

  /// Destructor
  ~Vector()
  {
    if (_bs != 1)
      MPI_Type_free(&_datatype);
  }

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Set all entries (including ghosts)
  /// @param[in] v The value to set all entries to (on calling rank)
  void set(T v) { std::fill(_x.begin(), _x.end(), v); }

  /// Begin scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_begin()
  {
    assert(_map);

    // Pack send buffer
    const std::vector<std::int32_t>& indices
        = _map->scatter_fwd_indices().array();
    _buffer_send_fwd.resize(_bs * indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
      std::copy_n(std::next(_x.cbegin(), _bs * indices[i]), _bs,
                  std::next(_buffer_send_fwd.begin(), _bs * i));
    }

    _buffer_recv_fwd.resize(_bs * _map->num_ghosts());
    _map->scatter_fwd_begin(xtl::span<const T>(_buffer_send_fwd), _datatype,
                            _request, xtl::span<T>(_buffer_recv_fwd));
  }

  /// End scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_end()
  {
    assert(_map);
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span xremote(_x.data() + local_size, _map->num_ghosts() * _bs);
    _map->scatter_fwd_end(_request);

    // Copy received data into ghost positions
    const std::vector<std::int32_t>& scatter_fwd_ghost_pos
        = _map->scatter_fwd_ghost_positions();
    for (std::size_t i = 0; i < _map->num_ghosts(); ++i)
    {
      const int pos = scatter_fwd_ghost_pos[i];
      std::copy_n(std::next(_buffer_recv_fwd.cbegin(), _bs * pos), _bs,
                  std::next(xremote.begin(), _bs * i));
    }
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
    // Pack send buffer
    const std::int32_t local_size = _bs * _map->size_local();
    xtl::span<const T> xremote(_x.data() + local_size,
                               _map->num_ghosts() * _bs);
    const std::vector<std::int32_t>& scatter_fwd_ghost_pos
        = _map->scatter_fwd_ghost_positions();
    _buffer_recv_fwd.resize(_bs * scatter_fwd_ghost_pos.size());
    for (std::size_t i = 0; i < scatter_fwd_ghost_pos.size(); ++i)
    {
      const int pos = scatter_fwd_ghost_pos[i];
      std::copy_n(std::next(xremote.cbegin(), _bs * i), _bs,
                  std::next(_buffer_recv_fwd.begin(), _bs * pos));
    }

    // Resize receive buffer and begin scatter
    _buffer_send_fwd.resize(_bs * _map->scatter_fwd_indices().array().size());
    _map->scatter_rev_begin(xtl::span<const T>(_buffer_recv_fwd), _datatype,
                            _request, xtl::span<T>(_buffer_send_fwd));
  }

  /// End scatter of ghost data to owner. This process may receive data
  /// from more than one process, and the received data can be summed or
  /// inserted into the local portion of the vector.
  /// @param op The operation to perform when adding/setting received
  /// values (add or insert)
  /// @note Collective MPI operation
  void scatter_rev_end(common::IndexMap::Mode op)
  {
    // Complete scatter
    _map->scatter_rev_end(_request);

    // Copy/accumulate into owned part of the vector
    const std::vector<std::int32_t>& shared_indices
        = _map->scatter_fwd_indices().array();
    switch (op)
    {
    case common::IndexMap::Mode::insert:
      for (std::size_t i = 0; i < shared_indices.size(); ++i)
      {
        std::copy_n(std::next(_buffer_send_fwd.cbegin(), _bs * i), _bs,
                    std::next(_x.begin(), _bs * shared_indices[i]));
      }
      break;
    case common::IndexMap::Mode::add:
      for (std::size_t i = 0; i < shared_indices.size(); ++i)
        for (int j = 0; j < _bs; ++j)
          _x[shared_indices[i] * _bs + j] += _buffer_send_fwd[i * _bs + j];
      break;
    }
  }

  /// Scatter ghost data to owner. This process may receive data from
  /// more than one process, and the received data can be summed or
  /// inserted into the local portion of the vector.
  /// @param op IndexMap operation (add or insert)
  /// @note Collective MPI operation
  void scatter_rev(dolfinx::common::IndexMap::Mode op)
  {
    this->scatter_rev_begin();
    this->scatter_rev_end(op);
  }

  /// Compute the norm of the vector
  /// @note Collective MPI operation
  /// @param type Norm type (supported types are \f$L^2\f$ and \f$L^\infty\f$)
  double norm(Norm type = Norm::l2) const
  {
    switch (type)
    {
    case Norm::l2:
      return std::sqrt(this->squared_norm());
    case Norm::linf:
    {
      const std::int32_t size_local = _bs * _map->size_local();
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
    const std::int32_t size_local = _bs * _map->size_local();
    double result = std::transform_reduce(
        _x.begin(), std::next(_x.begin(), size_local), double(0),
        std::plus<double>(), [](T val) { return std::norm(val); });
    double norm2;
    MPI_Allreduce(&result, &norm2, 1, MPI_DOUBLE, MPI_SUM, _map->comm());
    return norm2;
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// Get local part of the vector (const version)
  xtl::span<const T> array() const { return xtl::span<const T>(_x); }

  /// Get local part of the vector
  xtl::span<T> mutable_array() { return xtl::span(_x); }

  /// Get the allocator associated with the container
  constexpr allocator_type allocator() const { return _x.get_allocator(); }

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
  xtl::span<const T> x_a = a.array();
  xtl::span<const T> x_b = b.array();

  const T local = std::transform_reduce(
      x_a.begin(), std::next(x_a.begin(), local_size), x_b.begin(),
      static_cast<T>(0), std::plus<T>(),
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
                a.map()->comm());
  return result;
}

/// Orthonormalize a set of vectors
/// @param[in,out] basis The set of vectors to orthonormalise. The
/// vectors must have identical parallel layouts. The vectors are
/// modified in-place.
/// @param[in] tol The tolerance used to detect a linear dependency
template <typename T, typename U>
void orthonormalize(const xtl::span<Vector<T, U>>& basis, double tol = 1.0e-10)
{
  // Loop over each vector in basis
  for (std::size_t i = 0; i < basis.size(); ++i)
  {
    // Orthogonalize vector i with respect to previously orthonormalized
    // vectors
    for (std::size_t j = 0; j < i; ++j)
    {
      // basis_i <- basis_i - dot_ij  basis_j
      T dot_ij = inner_product(basis[i], basis[j]);
      std::transform(basis[j].array().begin(), basis[j].array().end(),
                     basis[i].array().begin(), basis[i].mutable_array().begin(),
                     [dot_ij](auto xj, auto xi) { return xi - dot_ij * xj; });
    }

    // Normalise basis function
    double norm = basis[i].norm(Norm::l2);
    if (norm < tol)
    {
      throw std::runtime_error(
          "Linear dependency detected. Cannot orthogonalize.");
    }
    std::transform(basis[i].array().begin(), basis[i].array().end(),
                   basis[i].mutable_array().begin(),
                   [norm](auto x) { return x / norm; });
  }
}

/// Test if basis is orthonormal
/// @param[in] basis The set of vectors to check
/// @param[in] tol The tolerance used to test for orthonormality
/// @return True is basis is orthonormal, otherwise false
template <typename T, typename U>
bool is_orthonormal(const xtl::span<const Vector<T, U>>& basis,
                    double tol = 1.0e-10)
{
  for (std::size_t i = 0; i < basis.size(); i++)
  {
    for (std::size_t j = i; j < basis.size(); j++)
    {
      const double delta_ij = (i == j) ? 1.0 : 0.0;
      T dot_ij = inner_product(basis[i], basis[j]);
      if (std::abs(delta_ij - dot_ij) > tol)
        return false;
    }
  }

  return true;
}

} // namespace dolfinx::la
