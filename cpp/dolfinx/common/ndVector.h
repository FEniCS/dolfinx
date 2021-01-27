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
  using const_reference = typename std::vector<T, Allocator>::const_reference;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using iterator = typename std::vector<T, Allocator>::iterator;
  using const_iterator = typename std::vector<T, Allocator>::const_iterator;

  ndVector() : rows_(0), cols_(0) { storage_ = std::vector<T, Allocator>(); }

  ndVector(size_type rows, size_type columns, value_type value = T(),
           const Allocator& alloc = Allocator())
      : rows_(rows), cols_(columns)
  {
    storage_ = std::vector<T, Allocator>(rows_ * cols_, value, alloc);
  }

  ndVector(size_type rows, size_type columns, std::initializer_list<T> list,
           const Allocator& alloc = Allocator())
      : rows_(rows), cols_(columns)
  {
    if (rows_ * cols_ != list.size())
      throw std::runtime_error("Dimension mismatch");
    storage_ = std::vector<T, Allocator>(list, alloc);
  }

  reference operator()(size_type x, size_type y)
  {
    return storage_[x * cols_ + y];
  }

  const_reference operator()(size_type x, size_type y) const
  {
    return storage_[x * cols_ + y];
  }

  tcb::span<value_type> row(int i) const
  {
    size_type offset = i * cols_;
    return tcb::span<value_type>(pointer(&storage_[0] + offset), cols_);
  }

  iterator begin() noexcept { return storage_.begin(); }

  const_iterator begin() const noexcept { return storage_.begin(); }

  iterator end() noexcept { return storage_.end(); }

  const_iterator end() const noexcept { return storage_.end(); }

  const_iterator cbegin() noexcept { return storage_.cbegin(); }

  const_iterator cend() noexcept { return storage_.cend(); }

  size_type size() const noexcept { return storage_.size(); }

  size_type cols() const noexcept { return cols_; }

  size_type rows() const noexcept { return rows_; }

  std::pair<size_type, size_type> shape() const noexcept
  {
    return {rows_, cols_};
  }

  bool empty() const noexcept { return storage_.empty(); }

  /// Returns a reference to the first element in the container.
  /// Calling front on an empty container is undefined.
  reference front() { return storage_.front(); }

  reference back() { return storage_.back(); }

protected:
  size_type rows_;
  size_type cols_;
  std::vector<T, Allocator> storage_;
};
} // namespace dolfinx::common