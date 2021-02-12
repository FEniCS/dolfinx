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

/// @todo Remove
template <typename T, typename = void>
struct is_2d_container : std::false_type
{
};

/// @todo Remove
template <typename T>
struct is_2d_container<T, std::void_t<decltype(std::declval<T>().data()),
                                      decltype(std::declval<T>().rows()),
                                      decltype(std::declval<T>().cols())>>
    : std::true_type
{
};

namespace dolfinx::common
{

/// This class provides a dynamic 2-dimensional row-wise array
/// data structure
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

  /// @todo Remove, used for copying in eigen array
  /// Constructs a two dimensional array with the copy of the contents
  /// of other two dimensional container
  template <class Container,
            typename
            = typename std::enable_if<is_2d_container<Container>::value>::type>
  array2d(Container& array)
      : shape({size_type(array.rows()), size_type(array.cols())})
  {
    _storage.resize(array.size());
    std::copy(array.data(), array.data() + array.size(), _storage.begin());
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
  constexpr tcb::span<value_type> row(int i)
  {
    size_type offset = i * shape[1];
    return tcb::span<value_type>(std::next(_storage.data(), offset), shape[1]);
  }

  /// Access a row in the array (const version)
  /// @param[in] i Row index
  /// @return Span of the row data
  constexpr tcb::span<const value_type> row(int i) const
  {
    size_type offset = i * shape[1];
    return tcb::span<const value_type>(std::next(_storage.data(), offset),
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
} // namespace dolfinx::common
