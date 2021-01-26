// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "span.hpp"
#include <iostream>
#include <vector>

namespace dolfinx::common
{
template <typename T, class Allocator = std::allocator<T>>
class ndVector
{
public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = typename std::vector<T, Allocator>::size_type;
  using reference = typename std::vector<T, Allocator>::reference;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using iterator = typename std::vector<T, Allocator>::iterator;
  using const_iterator = typename std::vector<T, Allocator>::const_iterator;

  ndVector(size_type rows, size_type columns, value_type value = T(),
           const Allocator& alloc = Allocator())
      : rows(rows), columns(columns)
  {
    storage = std::vector<T, Allocator>(rows * columns, value, alloc);
  }

  reference operator()(int x, int y) { return storage[x * columns + y]; }

  tcb::span<value_type> row(int i)
  {
    size_type offset = i * columns;
    return tcb::span<value_type>(pointer(&storage[0] + offset), columns);
  }

  iterator begin() noexcept { return storage.begin(); }

  const_iterator cbegin() noexcept { return storage.cbegin(); }

  iterator end() noexcept { return storage.end(); }

  const_iterator cend() noexcept { return storage.cend(); }

  size_type size() const noexcept { return storage.size(); }

  std::pair<size_type, size_type> shape() const noexcept
  {
    return {rows, columns};
  }

  bool empty() const noexcept { return storage.empty(); }

  /// Returns a reference to the first element in the container.
  /// Calling front on an empty container is undefined.
  reference front() { return storage.front(); }

  reference back() { return storage.back(); }

protected:
  size_type rows;
  size_type columns;
  std::vector<T, Allocator> storage;
};
} // namespace dolfinx::common