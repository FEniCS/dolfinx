// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
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

/// This class provides a dynamic 2-dimensional array structure.
/// The representation is strictly local, i.e. it is not parallel
/// aware.
template <typename T, class Allocator = std::allocator<T>>
class array_2d
{
public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = typename std::vector<T, Allocator>::size_type;
  using reference = typename std::vector<T, Allocator>::reference;
  using const_reference = typename std::vector<T, Allocator>::const_reference;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using iterator = typename std::vector<T, Allocator>::iterator;
  using const_iterator = typename std::vector<T, Allocator>::const_iterator;

  /// Construct an empty two dimensional array.
  array_2d() : _rows(0), _cols(0) { _storage = std::vector<T, Allocator>(); }

  array_2d(size_type rows, size_type columns, value_type value = T(),
           const Allocator& alloc = Allocator())
      : _rows(rows), _cols(columns)
  {
    _storage = std::vector<T, Allocator>(_rows * _cols, value, alloc);
  }

  // TODO: remvoe this construtctor.
  // Only used for
  template <class Container,
            typename
            = typename std::enable_if<is_2d_container<Container>::value>::type>
  array_2d(Container& array)
  {
    _rows = array.rows();
    _cols = array.cols();

    std::copy(array.data(), array.data() + array.size(),
              std::back_inserter(_storage));
  }

  array_2d(size_type rows, size_type columns, std::initializer_list<T> list,
           const Allocator& alloc = Allocator())
      : _rows(rows), _cols(columns)
  {
    if (_rows * _cols != list.size())
      throw std::runtime_error("Dimension mismatch");
    _storage = std::vector<T, Allocator>(list, alloc);
  }

  reference operator()(size_type x, size_type y)
  {
    return _storage[x * _cols + y];
  }

  const_reference operator()(size_type x, size_type y) const
  {
    return _storage[x * _cols + y];
  }

  tcb::span<value_type> row(int i) const
  {
    size_type offset = i * _cols;
    return tcb::span<value_type>(pointer(&_storage[0] + offset), _cols);
  }

  value_type* data() noexcept { return _storage.data(); }

  const value_type* data() const noexcept { return _storage.data(); };

  iterator begin() noexcept { return _storage.begin(); }

  const_iterator begin() const noexcept { return _storage.begin(); }

  iterator end() noexcept { return _storage.end(); }

  const_iterator end() const noexcept { return _storage.end(); }

  const_iterator cbegin() noexcept { return _storage.cbegin(); }

  const_iterator cend() noexcept { return _storage.cend(); }

  size_type size() const noexcept { return _storage.size(); }

  size_type cols() const noexcept { return _cols; }

  size_type rows() const noexcept { return _rows; }

  std::pair<size_type, size_type> shape() const noexcept
  {
    return {_rows, _cols};
  }

  void resize(size_type rows, size_type cols, value_type val = value_type())
  {
    // TODO: check rows and cols
    _storage.resize(rows * cols, val);
    _rows = rows;
    _cols = cols;
  }

  /// Could be improved with ndrange
  template <class Container,
            typename
            = typename std::enable_if<is_2d_container<Container>::value>::type>
  void copy(const Container& other)
  {
    std::int32_t rows = std::min<std::int32_t>(_rows, other.rows());
    std::int32_t cols = std::min<std::int32_t>(_cols, other.cols());
    for (std::int32_t i = 0; i < rows; i++)
    {
      for (std::int32_t j = 0; j < cols; j++)
        _storage[i * _cols + j] = other(i, j);
    }
  }

  bool empty() const noexcept { return _storage.empty(); }

  /// Returns a reference to the first element in the container.
  /// Calling front on an empty container is undefined.
  reference front() { return _storage.front(); }

  reference back() { return _storage.back(); }

protected:
  size_type _rows;
  size_type _cols;
  std::vector<T, Allocator> _storage;
};
} // namespace dolfinx::common