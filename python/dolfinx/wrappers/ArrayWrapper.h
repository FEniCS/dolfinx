// Copyright (C) 2020 Matthias Rambausek
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Core>
#include <memory>

namespace dolfinx_wrappers
{

// TODO: handle unique_ptr<Array> as well

/// Enables to return a shared_ptr to an Eigen Array from C++
template <typename Array>
class ArrayPtr
{
public:
  explicit ArrayPtr(std::shared_ptr<Array> array) : _array_ptr{std::move(array)} {}

  Array& get() const
  {
    assert(_array_ptr);
    return *_array_ptr;
  }

private:
  std::shared_ptr<Array> _array_ptr;
};

} // namespace dolfinx_wrappers
