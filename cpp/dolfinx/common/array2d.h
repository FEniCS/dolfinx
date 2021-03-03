// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
#include <array>
#include <cassert>
#include <numeric>
#include <vector>

template <typename T, typename = std::array<std::size_t, 2>>
struct has_shape : std::false_type
{
};

template <typename T>
struct has_shape<T, decltype(T::shape)> : std::true_type
{
};

namespace dolfinx
{

template <typename T, std::size_t N>
class ndspan;

/// This class provides a dynamic 2-dimensional row-wise array data
/// structure
template <typename T, std::size_t N, class Allocator = std::allocator<T>>
class ndarray
{
public:
  /// \cond DO_NOT_DOCUMENT
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = typename std::vector<T, Allocator>::size_type;
  using reference = typename std::vector<T, Allocator>::reference;
  using const_reference = typename std::vector<T, Allocator>::const_reference;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using iterator = typename std::vector<T, Allocator>::iterator;
  using const_iterator = typename std::vector<T, Allocator>::const_iterator;
  // using rank = N;
  /// \endcond

  /// Construct a two dimensional array
  /// @param[in] shape The shape the array {rows, cols}
  /// @param[in] value Initial value for all entries
  /// @param[in] alloc The memory allocator for the data storage
  ndarray(std::array<size_type, N> shape, value_type value = T(),
          const Allocator& alloc = Allocator())
      : shape(shape)
  {
    size_type size = std::accumulate(shape.begin(), shape.end(), 1,
                                     std::multiplies<size_type>());
    _storage = std::vector<T, Allocator>(size, value, alloc);
  }

  /// Construct a two dimensional array
  /// @param[in] rows The number of rows
  /// @param[in] cols The number of columns
  /// @param[in] value Initial value for all entries
  /// @param[in] alloc The memory allocator for the data storage
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  ndarray(size_type rows, size_type cols, value_type value = T(),
          const Allocator& alloc = Allocator())
      : shape({rows, cols})
  {
    _storage = std::vector<T, Allocator>(shape[0] * shape[1], value, alloc);
  }

  /// Constructs a two dimensional array from a vector
  template <typename Vector,
            typename
            = typename std::enable_if<std::is_class<Vector>::value>::type>
  ndarray(std::array<size_type, N> shape, Vector&& x)
      : shape(shape), _storage(std::forward<Vector>(x))
  {
    // Do nothing
  }

  /// @todo Decide what to do here
  /// Construct a two dimensional array using nested initializer lists
  /// @param[in] list The nested initializer list
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr ndarray(std::initializer_list<std::initializer_list<T>> list)
      : shape({list.size(), (*list.begin()).size()})
  {
    _storage.reserve(shape[0] * shape[1]);
    for (std::initializer_list<T> l : list)
      for (const T val : l)
        _storage.push_back(val);
  }

  /// Construct a two dimensional array from a two dimensional span
  /// @param[in] s The span
  template <typename Span2d,
            typename = typename std::enable_if<has_shape<Span2d>::value>>
  constexpr ndarray(Span2d& s)
      : shape(s.shape), _storage(s.data(), s.data() + s.shape[0] * s.shape[1])
  {
    // Do nothing
  }

  /// Copy constructor
  ndarray(const ndarray& x) = default;

  /// Move constructor
  ndarray(ndarray&& x) = default;

  /// Destructor
  ~ndarray() = default;

  /// Copy assignment
  ndarray& operator=(const ndarray& x) = default;

  /// Move assignment
  ndarray& operator=(ndarray&& x) = default;

  /// Return a reference to the element at specified location (i, j)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr reference operator()(size_type i, size_type j)
  {
    return _storage[i * shape[1] + j];
  }

  /// Return a reference to the element at specified location (i, j)
  /// (const version)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
  // template <typename std::enable_if<N == 2>::value>
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr const_reference operator()(size_type i, size_type j) const
  {
    return _storage[i * shape[1] + j];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr reference operator()(size_type i, size_type j, size_type k)
  {
    return _storage[shape[2] * (i * shape[1] + j) + k];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr const_reference operator()(size_type i, size_type j,
                                       size_type k) const
  {
    return _storage[shape[2] * (i * shape[1] + j) + k];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(std::next(_storage.data(), i * shape[1]),
                                 shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<const value_type> row(size_type i) const
  {
    return tcb::span<const value_type>(std::next(_storage.data(), i * shape[1]),
                                       shape[1]);
  }

  /// Get pointer to the first element of the underlying storage
  /// @warning Use this with caution - the data storage may be strided
  constexpr value_type* data() noexcept { return _storage.data(); }

  /// Get pointer to the first element of the underlying storage (const
  /// version)
  /// @warning Use this with caution - the data storage may be strided
  constexpr const value_type* data() const noexcept { return _storage.data(); };

  /// Returns the number of elements in the array
  /// @warning Use this caution - the data storage may be strided, i.e.
  /// the size of the underlying storage may be greater than
  /// sizeof(T)*(rows * cols)
  constexpr size_type size() const noexcept { return _storage.size(); }

  /// Returns the strides of the array
  template <int _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr std::array<size_type, 2> strides() const noexcept
  {
    return {shape[1] * sizeof(T), sizeof(T)};
  }

  /// Checks whether the container is empty
  /// @return Returns true if underlying storage is empty
  constexpr bool empty() const noexcept { return _storage.empty(); }

  /// Get the rank
  constexpr std::size_t rank() const noexcept { return N; }

  /// The shape of the array
  std::array<size_type, N> shape;

private:
  std::vector<T, Allocator> _storage;
};

template <typename T>
using array2d = ndarray<T, 2>;

/// This class provides a view into a 2-dimensional row-wise array of data
template <typename T, std::size_t N = 2>
class ndspan
{
public:
  // /// \cond DO_NOT_DOCUMENT
  using value_type = T;
  using size_type = std::size_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  // /// \endcond

  /// Construct a two dimensional array
  /// @param[in] data  pointer to the array to construct a view for
  /// @param[in] shape The shape the array {rows, cols}
  constexpr ndspan(T* data, std::array<size_type, N> shape)
      : _storage(data), shape(shape)
  {
    // Do nothing
  }

  /// Construct a two dimensional span from a two dimensional array
  template <typename Array2d,
            typename = typename std::enable_if<has_shape<Array2d>::value>>
  constexpr ndspan(Array2d& x) : shape(x.shape), _storage(x.data())
  {
    // Do nothing
  }

  /// Return a reference to the element at specified location (i, j)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr reference operator()(size_type i, size_type j)
  {
    return _storage[i * shape[1] + j];
  }

  /// Return a reference to the element at specified location (i, j)
  /// (const version)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr reference operator()(size_type i, size_type j) const
  {
    return _storage[i * shape[1] + j];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr reference operator()(size_type i, size_type j, size_type k)
  {
    return _storage[shape[2] * (i * shape[1] + j) + k];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr const_reference operator()(size_type i, size_type j,
                                       size_type k) const
  {
    return _storage[shape[2] * (i * shape[1] + j) + k];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(_storage + i * shape[1], shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<const value_type> row(size_type i) const
  {
    return tcb::span<const value_type>(_storage + i * shape[1], shape[1]);
  }

  /// Get pointer to the first element of the underlying storage
  /// @warning Use this with caution - the data storage may be strided
  // constexpr value_type* data() noexcept { return _storage; }

  /// Get pointer to the first element of the underlying storage (const
  /// version)
  /// @warning Use this with caution - the data storage may be strided
  constexpr value_type* data() const noexcept { return _storage; };

  /// Returns the number of elements in the span
  /// @warning Use this caution - the data storage may be strided, i.e.
  /// the size of the underlying storage may be greater than
  /// sizeof(T)*(rows * cols)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr size_type size() const noexcept
  {
    return shape[0] * shape[1];
  }

  /// Returns the strides of the array
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr std::array<size_type, 2> strides() const noexcept
  {
    return {shape[1] * sizeof(T), sizeof(T)};
  }

  /// Get the rank
  constexpr std::size_t rank() const noexcept { return N; }

  /// The shape of the array
  std::array<size_type, N> shape;

private:
  T* _storage;
};

template <typename T>
using span2d = ndspan<T, 2>;

} // namespace dolfinx
