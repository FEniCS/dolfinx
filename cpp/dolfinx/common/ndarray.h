// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <ostream>
#include <vector>

namespace dolfinx
{

template <typename T, typename = std::array<std::size_t, T::rank>>
struct has_shape : std::false_type
{
};

template <typename T>
struct has_shape<T, decltype(T::shape)> : std::true_type
{
};

template <typename T, std::size_t N>
class ndspan;

/// This class provides a dynamic n-dimensional row-wise array data
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
  /// \endcond

  /// Construct an n-dimensional array
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
    stride.back() = 1;
    std::partial_sum(shape.rbegin(), shape.rend() - 1, stride.rbegin() + 1,
                     std::multiplies<size_type>());
  }

  /// Construct an 2d-dimensional array
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
    stride.back() = 1;
    std::partial_sum(shape.rbegin(), shape.rend() - 1, stride.rbegin() + 1,
                     std::multiplies<size_type>());
  }

  /// Constructs an n-dimensional array from a vector
  template <typename Vector,
            typename = std::enable_if_t<std::is_class<Vector>::value>>
  ndarray(std::array<size_type, N> shape, Vector&& x)
      : shape(shape), _storage(std::forward<Vector>(x))
  {
    stride.back() = 1;
    std::partial_sum(shape.rbegin(), shape.rend() - 1, stride.rbegin() + 1,
                     std::multiplies<size_type>());
  }

  /// @todo Decide what to do here, perhaps remove and add assignment
  /// operator
  /// Construct an n-dimensional array using nested initializer lists
  /// @param[in] list The nested initializer list
  template <typename = std::enable_if_t<N == 2>>
  constexpr ndarray(std::initializer_list<std::initializer_list<T>> list)
      : shape({list.size(), (*list.begin()).size()})
  {
    _storage.reserve(shape[0] * shape[1]);
    for (std::initializer_list<T> l : list)
      for (const T val : l)
        _storage.push_back(val);
    stride.back() = 1;
    std::partial_sum(shape.rbegin(), shape.rend() - 1, stride.rbegin() + 1,
                     std::multiplies<size_type>());
  }

  /// @todo Need to handle strides correctly
  /// Construct an n-dimensional array from an n-dimensional span
  /// @param[in] s The span
  template <typename Span, typename = std::enable_if_t<has_shape<Span>::value>>
  constexpr ndarray(Span& s)
      : shape(s.shape), stride(s.stride),
        _storage(s.data(), s.data() + s.size())
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
    return _storage[i * stride[0] + j];
  }

  /// Return a reference to the element at specified location (i, j)
  /// (const version)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr const_reference operator()(size_type i, size_type j) const
  {
    return _storage[i * stride[0] + j];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr reference operator()(size_type i, size_type j, size_type k)
  {
    return _storage[i * stride[0] * j * stride[1] + k];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr const_reference operator()(size_type i, size_type j,
                                       size_type k) const
  {
    return _storage[i * stride[0] * j * stride[1] + k];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(std::next(_storage.data(), i * stride[0]),
                                 shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<const value_type> row(size_type i) const
  {
    return tcb::span<const value_type>(
        std::next(_storage.data(), i * stride[0]), shape[1]);
  }

  /// Access a row in the array
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr ndspan<value_type, 2> row(size_type i)
  {
    return ndspan<value_type, 2>(std::next(_storage.data(), i * stride[0]),
                                 {shape[1], shape[2]}, {stride[1], stride[2]});
  }

  /// Access a row in the array (const version)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr ndspan<const value_type, 2> row(size_type i) const
  {
    return ndspan<const value_type, 2>(
        std::next(_storage.data(), i * stride[0]), {shape[1], shape[2]},
        {stride[1], stride[2]});
  }

  /// Access a block in the array
  constexpr ndspan<value_type, N> block(std::array<size_type, N> start,
                                        std::array<size_type, N> shape)
  {
    size_type offset
        = std::inner_product(stride.begin(), stride.end(), start.begin(), 0);
    return ndspan<value_type, N>(_storage.data() + offset, shape, stride);
  }

  /// Access a block in the array (const)
  constexpr ndspan<value_type, N> block(std::array<size_type, N> start,
                                        std::array<size_type, N> shape) const
  {
    size_type offset
        = std::inner_product(stride.begin(), stride.end(), start.begin(), 0);
    return ndspan<value_type, N>(_storage.data() + offset, shape, stride);
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
  constexpr std::array<size_type, N> strides() const
  {
    std::array<size_type, N> s;
    std::transform(stride.begin(), stride.end(), s.begin(),
                   [](size_type c) { return sizeof(T) * c; });
    return s;
  }

  /// Checks whether the container is empty
  /// @return Returns true if underlying storage is empty
  constexpr bool empty() const noexcept { return _storage.empty(); }

  /// The shape of the array
  std::array<size_type, N> shape;

  /// Array strides. strides[i] is the numbers of values to needed to
  /// advance one values in the ith dimension
  std::array<size_type, N> stride;

  /// The rank of the array
  static constexpr size_type rank = size_type(N);

  /// Pretty printing, useful for debuging
  template <typename Array>
  friend std::ostream& operator<<(std::ostream& out, const Array& array);

private:
  std::vector<T, Allocator> _storage;
};

/// This class provides a view into an n-dimensional row-wise array of data
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

  /// Construct an n-dimensional array
  /// @param[in] data  pointer to the array to construct a view for
  /// @param[in] shape The shape the array {rows, cols}
  constexpr ndspan(T* data, std::array<size_type, N> shape)
      : _storage(data), shape(shape)
  {
    stride.back() = 1;
    std::partial_sum(shape.rbegin(), shape.rend() - 1, stride.rbegin() + 1,
                     std::multiplies<size_type>());
  }

  /// Construct an n-dimensional array
  /// @param[in] data  pointer to the array to construct a view for
  /// @param[in] shape The shape the array {rows, cols}
  /// @param[in] stride The array stides (by entries, not bytes)
  constexpr ndspan(T* data, std::array<size_type, N> shape,
                   std::array<size_type, N> stride)
      : _storage(data), shape(shape), stride(stride)
  {
    assert(stride.back() == 1);
  }

  /// Construct an n-dimensional span from an n-dimensional array
  template <typename Array,
            typename = std::enable_if_t<has_shape<Array>::value>>
  constexpr ndspan(Array& x)
      : shape(x.shape), stride(x.stride), _storage(x.data())
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
    return _storage[i * stride[0] + j];
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
    return _storage[i * stride[0] + j];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr reference operator()(size_type i, size_type j, size_type k)
  {
    return _storage[i * stride[0] + j * stride[1] + k];
  }

  /// Return a reference to the element at specified location (i, j, k)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr const_reference operator()(size_type i, size_type j,
                                       size_type k) const
  {
    return _storage[i * stride[0] + j * stride[1] + k];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(_storage + i * stride[0], shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 2>>
  constexpr tcb::span<const value_type> row(size_type i) const
  {
    return tcb::span<const value_type>(_storage + i * stride[0], shape[1]);
  }

  /// Access a row in the array
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr ndspan<value_type, 2> row(size_type i)
  {
    return ndspan<value_type, 2>(_storage + i * stride[0],
                                 {shape[1], shape[2]});
  }

  /// Access a row in the array (const version)
  template <std::size_t _N = N, typename = std::enable_if_t<_N == 3>>
  constexpr ndspan<const value_type, 2> row(size_type i) const
  {
    return ndspan<const value_type, 2>(_storage + i * stride[0],
                                       {shape[1], shape[2]});
  }

  /// Access a block in the array
  constexpr ndspan<value_type, N> block(std::array<size_type, N> start,
                                        std::array<size_type, N> shape)
  {
    size_type offset
        = std::inner_product(stride.begin(), stride.end(), start.begin(), 0);
    return ndspan<value_type, 2>(_storage + offset, shape, stride);
  }

  /// Access a block in the array (const)
  constexpr ndspan<value_type, N> block(std::array<size_type, N> start,
                                        std::array<size_type, N> shape) const
  {
    size_type offset
        = std::inner_product(stride.begin(), stride.end(), start.begin(), 0);
    return ndspan<value_type, 2>(_storage + offset, shape, stride);
  }

  /// Get pointer to the first element of the underlying storage
  /// @warning Use this with caution - the data storage may be strided
  // constexpr value_type* data() noexcept { return _storage; }

  /// Get pointer to the first element of the underlying storage (const
  /// version)
  /// @warning Use this with caution - the data storage may be strided
  constexpr T* data() const noexcept { return _storage; };

  /// Returns the number of elements in the span
  /// @warning Use this caution - the data storage may be strided, i.e.
  /// the size of the underlying storage may be greater than
  /// sizeof(T) * (rows * cols)
  constexpr size_type size() const
  {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_type>());
  }

  /// Returns the strides of the span in bytes
  constexpr std::array<size_type, 2> strides() const
  {
    std::array<size_type, N> s;
    std::transform(stride.begin(), stride.end(), s.begin(),
                   [](size_type c) { return sizeof(T) * c; });
    return s;
  }

  /// The shape of the span
  std::array<size_type, N> shape;

  /// Span strides. strides[i] is the numbers of values to needed to
  /// advance one values in the ith dimension.
  std::array<size_type, N> stride;

  /// The rank of the span
  static constexpr size_type rank = size_type(N);

  /// Pretty printing, useful for debuging
  template <typename Span>
  friend std::ostream& operator<<(std::ostream& out, const Span& array);

private:
  T* _storage;
};

/// Convenience function for outputting arrays
template <typename Array>
std::ostream& print_array(std::ostream& out, const Array& array)
{
  if constexpr (array.rank == 2)
    for (std::size_t i = 0; i < array.shape[0]; i++)
    {
      out << "{";
      for (std::size_t j = 0; j < array.shape[1]; j++)
        out << array(i, j) << ", ";
      out << "}" << std::endl;
    }

  if constexpr (array.rank == 3)
    for (std::size_t i = 0; i < array.shape[0]; i++)
    {
      for (std::size_t j = 0; j < array.shape[1]; j++)
      {
        out << "{";
        for (std::size_t k = 0; k < array.shape[2]; k++)
          out << array(i, j, k) << ", ";
        out << "}" << std::endl;
      }
      out << std::endl;
    }

  return out;
}

/// Pretty printing, useful for debuging
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const ndarray<T, N>& array)
{
  return print_array(out, array);
}

/// Pretty printing, useful for debuging
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const ndspan<T, N>& span)
{
  return print_array(out, span);
}

} // namespace dolfinx
