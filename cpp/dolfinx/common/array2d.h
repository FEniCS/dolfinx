// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
#include <array>
#include <cassert>
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

template <typename T>
class span2d;

/// This class provides a dynamic 2-dimensional row-wise array data
/// structure
template <typename T, class Allocator = std::allocator<T>>
class array2d
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

  /// Construct a two dimensional array
  /// @param[in] shape The shape the array {rows, cols}
  /// @param[in] value Initial value for all entries
  /// @param[in] alloc The memory allocator for the data storage
  array2d(std::array<size_type, 2> shape, value_type value = T(),
          const Allocator& alloc = Allocator())
      : shape(shape)
  {
    _storage = std::vector<T, Allocator>(shape[0] * shape[1], value, alloc);
  }

  /// Construct a two dimensional array
  /// @param[in] rows The number of rows
  /// @param[in] cols The number of columns
  /// @param[in] value Initial value for all entries
  /// @param[in] alloc The memory allocator for the data storage
  array2d(size_type rows, size_type cols, value_type value = T(),
          const Allocator& alloc = Allocator())
      : shape({rows, cols})
  {
    _storage = std::vector<T, Allocator>(shape[0] * shape[1], value, alloc);
  }

  /// Constructs a two dimensional array from a vector
  template <typename Vector,
            typename
            = typename std::enable_if<std::is_class<Vector>::value>::type>
  array2d(std::array<size_type, 2> shape, Vector&& x)
      : shape(shape), _storage(std::forward<Vector>(x))
  {
    // Do nothing
  }

  /// Construct a two dimensional array using nested initializer lists
  /// @param[in] list The nested initializer list
  constexpr array2d(std::initializer_list<std::initializer_list<T>> list)
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
  constexpr array2d(Span2d& s)
      : shape(s.shape), _storage(s.data(), s.data() + s.shape[0] * s.shape[1])
  {
    // Do nothing
  }

  /// Copy constructor
  array2d(const array2d& x) = default;

  /// Move constructor
  array2d(array2d&& x) = default;

  /// Destructor
  ~array2d() = default;

  /// Copy assignment
  array2d& operator=(const array2d& x) = default;

  /// Move assignment
  array2d& operator=(array2d&& x) = default;

  /// Return a reference to the element at specified location (i, j)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
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
  constexpr const_reference operator()(size_type i, size_type j) const
  {
    return _storage[i * shape[1] + j];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(std::next(_storage.data(), i * shape[1]),
                                 shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
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
  constexpr std::array<size_type, 2> strides() const noexcept
  {
    return {shape[1] * sizeof(T), sizeof(T)};
  }

  /// Checks whether the container is empty
  /// @return Returns true if underlying storage is empty
  constexpr bool empty() const noexcept { return _storage.empty(); }

  /// The shape of the array
  std::array<size_type, 2> shape;

private:
  std::vector<T, Allocator> _storage;
};

/// This class provides a view into a 2-dimensional row-wise array of data
template <typename T>
class span2d
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
  span2d(T* data, std::array<size_type, 2> shape) : _storage(data), shape(shape)
  {
    // Do nothing
  }

  /// Construct a two dimensional span from a two dimensional array
  template <typename Array2d,
            typename = typename std::enable_if<has_shape<Array2d>::value>>
  span2d(Array2d& x) : shape(x.shape), _storage(x.data())
  {
    // Do nothing
  }

  /// Return a reference to the element at specified location (i, j)
  /// @param[in] i Row index
  /// @param[in] j Column index
  /// @return Reference to the (i, j) item
  /// @note No bounds checking is performed
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
  constexpr reference operator()(size_type i, size_type j) const
  {
    return _storage[i * shape[1] + j];
  }

  /// Access a row in the array
  /// @param[in] i Row index
  /// @return Span of the row data
  constexpr tcb::span<value_type> row(size_type i)
  {
    return tcb::span<value_type>(_storage + i * shape[1], shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  constexpr tcb::span<const value_type> row(size_type i) const
  {
    return tcb::span<const value_type>(_storage + i * shape[1], shape[1]);
  }

  /// Get pointer to the first element of the underlying storage
  /// @warning Use this with caution - the data storage may be strided
  constexpr value_type* data() noexcept { return _storage; }

  /// Get pointer to the first element of the underlying storage (const
  /// version)
  /// @warning Use this with caution - the data storage may be strided
  constexpr const value_type* data() const noexcept { return _storage; };

  /// Returns the number of elements in the span
  /// @warning Use this caution - the data storage may be strided, i.e.
  /// the size of the underlying storage may be greater than
  /// sizeof(T)*(rows * cols)
  constexpr size_type size() const noexcept { return shape[0] * shape[1]; }

  /// The shape of the array
  std::array<size_type, 2> shape;

private:
  T* _storage;
};

} // namespace dolfinx
