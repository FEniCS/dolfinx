// Copyright (C) 2020-2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <algorithm>
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
/// @brief la::Vector scatter pack/unpack function concept.
template <class F, class Container, class ScatterContainer>
concept VectorPackKernel = requires(F f, ScatterContainer idx, Container x) {
  f(idx.cbegin(), idx.cend(), x.cbegin(), x.begin());
};

/// @brief Access to pointer function concept.
template <class GetPtr, class U, class T>
concept GetPtrConcept = requires(GetPtr f, U x) {
  { f(x) } -> std::same_as<T*>;
} and requires(GetPtr f, const U x) {
  { f(x) } -> std::same_as<const T*>;
};

/// @brief A vector that can be distributed across processes.
///
/// @tparam T Scalar type of the vector.
/// @tparam Container Data container type. This is typically
/// `std::vector<T>` on CPUs, and `thrust::device_vector<T>` on GPUs.
/// @tparam ScatterContainer Storage container type for the scatterer
/// indices. This is typically `std::vector<std::int32_t>` on CPUs, and
/// `thrust::device_vector<std::int32_t>` on GPUs.
template <typename T, typename Container = std::vector<T>,
          typename ScatterContainer = std::vector<std::int32_t>>
class Vector
{
  static_assert(std::is_same_v<typename Container::value_type, T>);

  template <typename, typename, typename>
  friend class Vector;

private:
  /// @brief Return a 'pack' function for packing a send buffer.
  ///
  /// Typically used for forward and reverse scatter operations on a
  /// CPU.
  auto get_pack()
  {
    return [](typename ScatterContainer::const_iterator idx_first,
              typename ScatterContainer::const_iterator idx_last,
              const auto in_first, auto out_first)
    {
      // out[i] = in[idx[i]]
      std::transform(idx_first, idx_last, out_first,
                     [in_first](auto p) { return *std::next(in_first, p); });
    };
  }

  /// @brief Return an 'unpack' function for unpacking a receive buffer. Assigns
  /// received values to entries.
  ///
  /// Typically used to unpack into ghost entries on a CPU.
  auto get_unpack()
  {
    return [](typename ScatterContainer::const_iterator idx_first,
              typename ScatterContainer::const_iterator idx_last,
              const auto in_first, auto out_first)
    {
      // out[idx[i]] = in[i]
      for (typename ScatterContainer::const_iterator idx = idx_first;
           idx != idx_last; ++idx)
      {
        std::size_t d = std::distance(idx_first, idx);
        *std::next(out_first, *idx) = *std::next(in_first, d);
      }
    };
  }

  /// @brief Return an 'unpack' function for unpacking a receive buffer.
  /// Applied a binary operation an then assigns to entries.
  ///
  /// Typically used to unpack into owned entries on a CPU. Commonly
  /// more than one value per entry is received.
  template <typename BinaryOp>
  auto get_unpack_op(BinaryOp op)
  {
    return [op](typename ScatterContainer::const_iterator idx_first,
                typename ScatterContainer::const_iterator idx_last,
                const auto in_first, auto out_first)
    {
      // out[idx[i]] = op(out[idx[i]], in[i])
      for (typename ScatterContainer::const_iterator idx = idx_first;
           idx != idx_last; ++idx)
      {
        std::size_t d = std::distance(idx_first, idx);
        auto& out = *std::next(out_first, *idx);
        out = op(out, *std::next(in_first, d));
      }
    };
  }

public:
  /// Container type
  using container_type = Container;

  /// Scalar type
  using value_type = container_type::value_type;

  static_assert(std::is_same_v<value_type, typename container_type::value_type>,
                "Scalar type and container value type must be the same.");

  /// @brief Create a distributed vector.
  ///
  /// @param map Index map that describes the parallel layout of
  /// the data.
  /// @param bs Number of entries per index map 'index' (block size).
  Vector(std::shared_ptr<const common::IndexMap> map, int bs)
      : _map(map), _bs(bs), _x(bs * (map->size_local() + map->num_ghosts())),
        _scatterer(
            std::make_shared<common::Scatterer<ScatterContainer>>(*_map, bs)),
        _buffer_local(_scatterer->local_indices().size()),
        _buffer_remote(_scatterer->remote_indices().size())
  {
  }

  /// Copy constructor
  Vector(const Vector& x) = default;

  /// Move constructor
  Vector(Vector&& x) = default;

private:
  /// @brief Return a shared pointer to a Scatter, which points (1) to
  /// the input Scatterer or (2) to a copy of input Scatterer.
  ///
  /// If the new and old Vectors share the same Scatterer type, the
  /// Scatter can be shared. If the new Vector uses a different
  /// Scatterer storage type, then the Scatterer needs to be copied.
  ///
  /// @param sc Scatter of the Vector being copied.
  /// @return Scatter for use with the new Vector.
  std::shared_ptr<const common::Scatterer<ScatterContainer>>
  scatter_ptr(auto sc) const
  {
    using SC = typename std::remove_cv<
        typename decltype(sc)::element_type>::type::container_type;
    if constexpr (std::is_same_v<ScatterContainer, SC>)
      return sc; // Scatters use same container
    else         // Scatters use different containers, so copy
      return std::make_shared<common::Scatterer<ScatterContainer>>(*sc);
  }

public:
  /// @brief Copy-convert vector, possibly using to different container
  /// types.
  ///
  /// Examples of use include copying a Vector to a different value
  /// type, e.g. double to float, or copying a Vector from a CPU to a
  /// GPU.
  ///
  /// @tparam Vec Type of the Vector being copied.
  /// @param x Vector to copy.
  template <typename T0, typename Container0, typename ScatterContainer0>
  explicit Vector(const Vector<T0, Container0, ScatterContainer0>& x)
      : _map(x.index_map()), _bs(x.bs()), _x(x._x.begin(), x._x.end()),
        _scatterer(scatter_ptr(x._scatterer)), _request(MPI_REQUEST_NULL),
        _buffer_local(_scatterer->local_indices().size()),
        _buffer_remote(_scatterer->remote_indices().size())
  {
  }

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move assignment operator
  Vector& operator=(Vector&& x) = default;

  /// @deprecated Use `std::ranges::fill(u.array(), v)` instead.
  /// @brief Set all entries (including ghosts).
  ///
  /// @param[in] v Value to set all entries to (on calling rank).
  [[deprecated("Use std::ranges::fill(u.array(), v) instead.")]] void
  set(value_type v)
  {
    std::ranges::fill(_x, v);
  }

  /// @brief Begin scatter (send) of local data that is ghosted on other
  /// processes.
  ///
  /// The user provides the function to pack to the send buffer.
  /// Typically usage would be a specialised function to pack data that
  /// resides on a GPU.
  ///
  /// @note Collective MPI operation
  ///
  /// @tparam U Pack function type.
  /// @tparam GetPtr
  /// @param pack Function that packs owned data into a send buffer.
  /// @param get_ptr Function that for a `Container` type returns the
  /// pointer to the underlying data.
  template <typename U, typename GetPtr>
    requires VectorPackKernel<U, container_type, ScatterContainer>
             and GetPtrConcept<GetPtr, Container, T>
  void scatter_fwd_begin(U pack, GetPtr get_ptr)
  {
    pack(_scatterer->local_indices().begin(), _scatterer->local_indices().end(),
         _x.begin(), _buffer_local.begin());
    _scatterer->scatter_fwd_begin(get_ptr(_buffer_local),
                                  get_ptr(_buffer_remote), _request);
  }

  /// @brief Begin scatter (send) of local data that is ghosted on other
  /// processes (simplified CPU version).
  ///
  /// Suitable for scatter operations on a CPU with `std::vector`
  /// storage. The send buffer is packed internally by a function that
  /// is suitable for use on a CPU.
  ///
  /// @note Collective MPI operation.
  void scatter_fwd_begin()
    requires requires(Container c) {
      { c.data() } -> std::same_as<T*>;
    }
  {
    scatter_fwd_begin(get_pack(), [](auto&& x) { return x.data(); });
  }

  /// @brief End scatter (send) of local data values that are ghosted on
  /// other processes.
  ///
  /// The user provides the function to unpack the receive buffer.
  /// Typically usage would be a specialised function to unpack data
  /// that resides on a GPU.
  ///
  /// @note Collective MPI operation.
  ///
  /// @param unpack Function to unpack the receive buffer into the ghost
  /// entries.
  template <typename U>
    requires VectorPackKernel<U, container_type, ScatterContainer>
  void scatter_fwd_end(U unpack)
  {
    _scatterer->scatter_end(_request);
    unpack(_scatterer->remote_indices().begin(),
           _scatterer->remote_indices().end(), _buffer_remote.begin(),
           std::next(_x.begin(), _bs * _map->size_local()));
  }

  /// @brief End scatter (send) of local data values that are ghosted on
  /// other processes (simplified CPU version).
  ///
  /// Suitable for scatter operations on a CPU with `std::vector`
  /// storage. The receive buffer is unpacked internally by a function
  /// that is suitable for use on a CPU.
  ///
  /// @note Collective MPI operation.
  void scatter_fwd_end()
    requires requires(Container c) {
      { c.data() } -> std::same_as<T*>;
    }
  {
    this->scatter_fwd_end(get_unpack());
  }

  /// @brief Scatter (send) of local data values that are ghosted on
  /// other processes and update ghost entry values (simplified CPU
  /// version).
  ///
  /// Suitable for scatter operations on a CPU with `std::vector`
  /// storage. The send buffer is packed and the receive buffer unpacked
  /// by a function that is suitable for use on a CPU.
  ///
  /// @note Collective MPI operation
  void scatter_fwd()
    requires requires(Container c) {
      { c.data() } -> std::same_as<T*>;
    }
  {
    this->scatter_fwd_begin(get_pack(), [](auto&& x) { return x.data(); });
    this->scatter_fwd_end(get_unpack());
  }

  /// @brief Start scatter (send) of ghost entry data to the owning
  /// process of an index.
  ///
  /// The user provides the function to pack to the send buffer.
  /// Typically usage would be a specialised function to pack data that
  /// resides on a GPU.
  ///
  /// @note Collective MPI operation
  /// @tparam U Pack function type.
  /// @tparam GetPtr
  /// @param pack Function that packs ghost data into a send buffer.
  /// @param get_ptr Function that for a `Container` type returns the
  /// pointer to the underlying data.
  template <typename U, typename GetPtr>
    requires VectorPackKernel<U, container_type, ScatterContainer>
             and GetPtrConcept<GetPtr, Container, T>
  void scatter_rev_begin(U pack, GetPtr get_ptr)
  {
    std::int32_t local_size = _bs * _map->size_local();
    pack(_scatterer->remote_indices().begin(),
         _scatterer->remote_indices().end(), std::next(_x.begin(), local_size),
         _buffer_remote.begin());
    _scatterer->scatter_rev_begin(get_ptr(_buffer_remote),
                                  get_ptr(_buffer_local), _request);
  }

  /// @brief Start scatter (send) of ghost entry data to the owning
  /// process of an index (simplified CPU version).
  ///
  /// Suitable for scatter operations on a CPU with `std::vector`
  /// storage. The send buffer is packed internally by a function that
  /// is suitable for use on a CPU.
  ///
  /// @note Collective MPI operation
  void scatter_rev_begin()
    requires requires(Container c) {
      { c.data() } -> std::same_as<T*>;
    }
  {
    scatter_rev_begin(get_pack(), [](auto&& x) { return x.data(); });
  }

  /// @brief End scatter of ghost data to owner and update owned entries
  ///
  /// For an owned entry, data from more than one process may be
  /// received. The received data can be summed or inserted into the
  /// owning entry by the `unpack` function.
  ///
  /// @note Collective MPI operation
  template <typename U>
    requires VectorPackKernel<U, container_type, ScatterContainer>
  void scatter_rev_end(U unpack)
  {
    _scatterer->scatter_end(_request);
    unpack(_scatterer->local_indices().begin(),
           _scatterer->local_indices().end(), _buffer_local.begin(),
           _x.begin());
  }

  /// @brief Scatter (send) of ghost data values to the owning process
  /// and assign/accumulate into the owned data entries (simplified CPU
  /// version).
  ///
  /// For an owned entry, data from more than one process may be
  /// received. The received data can be summed or inserted into the
  /// owning entry. The this is controlled by the `op` function.
  ///
  /// @note Collective MPI operation
  ///
  /// @param op IndexMap operation (add or insert).
  template <class BinaryOperation>
    requires requires(Container c) {
      { c.data() } -> std::same_as<T*>;
    }
  void scatter_rev(BinaryOperation op)
  {
    this->scatter_rev_begin();
    this->scatter_rev_end(get_unpack_op(op));
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> index_map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// @brief Get the process-local part of the vector.
  ///
  /// Owned entries appear first, followed by ghosted entries.
  container_type& array() { return _x; }

  /// @brief Get the process-local part of the vector (const version).
  ///
  /// Owned entries appear first, followed by ghosted entries.
  const container_type& array() const { return _x; }

  /// @deprecated Use ::array instead.
  /// @brief Get local part of the vector
  [[deprecated("Use array() instead.")]] container_type& mutable_array()
  {
    return _x;
  }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  // Vector data
  container_type _x;

  // Scatter for managing MPI communication
  std::shared_ptr<const common::Scatterer<ScatterContainer>> _scatterer;

  // MPI request handle
  MPI_Request _request = MPI_REQUEST_NULL;

  // Buffers for ghost scatters
  container_type _buffer_local, _buffer_remote;
}; // namespace dolfinx::la

/// @brief Compute the inner product of two vectors.
///
/// Computes `a^{H} b` (`a^{T} b` if `a` and `b` are real). The two
/// vectors must have the same parallel layout.
///
/// @note Collective MPI operation
///
/// @param a Vector `a`.
/// @param b Vector `b`.
/// @return Inner product between `a` and `b`.
template <class V>
auto inner_product(const V& a, const V& b)
{
  using T = typename V::value_type;
  const std::int32_t local_size = a.bs() * a.index_map()->size_local();
  if (local_size != b.bs() * b.index_map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");

  const T local = std::transform_reduce(
      a.array().begin(), std::next(a.array().begin(), local_size),
      b.array().begin(), static_cast<T>(0), std::plus{},
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
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM,
                a.index_map()->comm());
  return result;
}

/// @brief Compute the squared L2 norm of vector.
///
/// @note Collective MPI operation.
template <class V>
auto squared_norm(const V& a)
{
  using T = typename V::value_type;
  T result = inner_product(a, a);
  return std::real(result);
}

/// @brief Compute the norm of the vector.
///
/// @note Collective MPI operation.
///
/// @param x Vector to compute the norm of.
/// @param type Norm type.
template <class V>
auto norm(const V& x, Norm type = Norm::l2)
{
  using T = typename V::value_type;
  switch (type)
  {
  case Norm::l1:
  {
    std::int32_t size_local = x.bs() * x.index_map()->size_local();
    using U = typename dolfinx::scalar_value_t<T>;
    U local_l1 = std::accumulate(
        x.array().begin(), std::next(x.array().begin(), size_local), U(0),
        [](auto norm, auto x) { return norm + std::abs(x); });
    U l1(0);
    MPI_Allreduce(&local_l1, &l1, 1, MPI::mpi_t<U>, MPI_SUM,
                  x.index_map()->comm());
    return l1;
  }
  case Norm::l2:
    return std::sqrt(squared_norm(x));
  case Norm::linf:
  {
    std::int32_t size_local = x.bs() * x.index_map()->size_local();
    auto max_pos = std::max_element(
        x.array().begin(), std::next(x.array().begin(), size_local),
        [](T a, T b) { return std::norm(a) < std::norm(b); });
    auto local_linf = std::abs(*max_pos);
    decltype(local_linf) linf = 0;
    MPI_Allreduce(&local_linf, &linf, 1, MPI::mpi_t<decltype(linf)>, MPI_MAX,
                  x.index_map()->comm());
    return linf;
  }
  default:
    throw std::runtime_error("Norm type not supported");
  }
}

/// @brief Orthonormalize a set of vectors.
///
/// @tparam V dolfinx::la::Vector
/// @param[in,out] basis The set of vectors to orthonormalise. The
/// vectors must have identical parallel layouts. The vectors are
/// modified in-place.
template <class V>
void orthonormalize(std::vector<std::reference_wrapper<V>> basis)
{
  using T = typename V::value_type;
  using U = typename dolfinx::scalar_value_t<T>;

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
      std::ranges::transform(bj.array(), bi.array(), bi.array().begin(),
                             [dot_ij](auto xj, auto xi)
                             { return xi - dot_ij * xj; });
    }

    // Normalise basis function
    auto norm = la::norm(bi, la::Norm::l2);
    if (norm * norm < std::numeric_limits<U>::epsilon())
    {
      throw std::runtime_error(
          "Linear dependency detected. Cannot orthogonalize.");
    }
    std::ranges::transform(bi.array(), bi.array().begin(),
                           [norm](auto x) { return x / norm; });
  }
}

/// @brief Test if basis is orthonormal.
///
/// Returns true if ||x_i - x_j|| - delta_{ij} < eps for all i, j, and
/// otherwise false.
///
/// @param[in] basis Set of vectors to check.
/// @param[in] eps Tolerance.
/// @return True is basis is orthonormal, otherwise false.
template <class V>
bool is_orthonormal(
    std::vector<std::reference_wrapper<const V>> basis,
    dolfinx::scalar_value_t<typename V::value_type> eps = std::numeric_limits<
        dolfinx::scalar_value_t<typename V::value_type>>::epsilon())
{
  using T = typename V::value_type;
  for (std::size_t i = 0; i < basis.size(); i++)
  {
    for (std::size_t j = i; j < basis.size(); ++j)
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
