// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
#include <cassert>
#include <vector>

template <typename T, typename = void>
struct is_2d_container : std::false_type
{
};

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
/// data structure. The representation is strictly local, i.e.
/// it is not parallel aware.
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

  array2d() = default;

  /// Constructs a two dimensional array with size = rows * cols, and initialize
  /// elements with value value.
  array2d(size_type rows, size_type cols, value_type value = T(),
          const Allocator& alloc = Allocator())
      : _rows(rows), _cols(cols)
  {
    _storage = std::vector<T, Allocator>(_rows * _cols, value, alloc);
  }

  /// Constructs a two dimensional array with the copy of the contents of other
  /// two dimensional container.
  template <class Container,
            typename
            = typename std::enable_if<is_2d_container<Container>::value>::type>
  array2d(Container& array)
  {
    _rows = array.rows();
    _cols = array.cols();
    _storage.resize(array.size());
    std::copy(array.data(), array.data() + array.size(), _storage.begin());
  }

  /// Constructs a two dimensional array with the contents of the initializer
  /// list init.
  array2d(size_type rows, size_type columns, std::initializer_list<T> list,
          const Allocator& alloc = Allocator())
      : _rows(rows), _cols(columns)
  {
    if (_rows * _cols != list.size())
      throw std::runtime_error("Dimension mismatch");
    _storage = std::vector<T, Allocator>(list, alloc);
  }

  /// Returns a reference to the element at specified location (i, j).
  /// No bounds checking is performed.
  reference operator()(size_type i, size_type j)
  {
    return _storage[i * _cols + j];
  }

  /// Returns a const reference to the element at specified location (i, j).
  /// No bounds checking is performed.
  const_reference operator()(size_type i, size_type j) const
  {
    return _storage[i * _cols + j];
  }

  /// Returns a reference to the row at specified location (i).
  tcb::span<const value_type> row(int i) const
  {
    size_type offset = i * _cols;
    return tcb::span<const value_type>(std::next(_storage.data(), offset),
                                       _cols);
  }

  /// Returns a pointer to the first element of the array.
  value_type* data() noexcept { return _storage.data(); }

  /// Returns a const pointer to the first element of the array.
  const value_type* data() const noexcept { return _storage.data(); };

  /// Returns an iterator to the first element of the array.
  iterator begin() noexcept { return _storage.begin(); }

  /// Returns an iterator to the first element of the array.
  const_iterator begin() const noexcept { return _storage.begin(); }

  /// Returns an iterator to the element following the last element.
  iterator end() noexcept { return _storage.end(); }

  /// Returns an iterator to the element following the last element.
  const_iterator end() const noexcept { return _storage.end(); }

  /// Returns a const iterator to the first element of the array.
  const_iterator cbegin() noexcept { return _storage.cbegin(); }

  /// Returns a const iterator to the element following the last element.
  const_iterator cend() noexcept { return _storage.cend(); }

  /// Returns the number of elements in the array (rows * cols).
  size_type size() const noexcept { return _storage.size(); }

  /// Returns the number of cols in the two-dimensional array.
  size_type cols() const noexcept { return _cols; }

  /// Returns the number of rows in the two-dimensional array.
  size_type rows() const noexcept { return _rows; }

  /// Returns the array dimensions {cols, rows}.
  std::pair<size_type, size_type> shape() const noexcept
  {
    return {_rows, _cols};
  }

  /// Changes the number of elements stored.
  void resize(size_type rows, size_type cols, value_type val = value_type())
  {
    // TODO: check rows and cols
    _rows = rows;
    _cols = cols;
    _storage.resize(rows * cols, val);
  }

  /// Copies a block of data from other container.
  template <class Container,
            typename
            = typename std::enable_if<is_2d_container<Container>::value>::type>
  void assign(const Container& other)
  {
    size_type rows = std::min<size_type>(_rows, other.rows());
    size_type cols = std::min<size_type>(_cols, other.cols());
    for (size_type i = 0; i < rows; i++)
    {
      for (size_type j = 0; j < cols; j++)
        _storage[i * _cols + j] = other(i, j);
    }
  }

  /// Checks whether the container is empty.
  bool empty() const noexcept { return _storage.empty(); }

private:
  size_type _rows;
  size_type _cols;
  std::vector<T, Allocator> _storage;
};
} // namespace dolfinx::common