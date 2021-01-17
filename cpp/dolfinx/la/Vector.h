// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Core>
#include <dolfinx/common/IndexMap.h>
#include <memory>

namespace dolfinx::la
{

/// Distributed vector

template <typename T>
class Vector
{
public:
  /// Create vector
  Vector(const std::shared_ptr<const common::IndexMap>& map, int bs)
      : _map(map), _bs(bs)
  {
    assert(map);
    const std::int32_t local_size
        = bs * (map->size_local() + map->num_ghosts());
    _x.resize(local_size);
  }

  /// Copy constructor
  Vector(const Vector& x) = default;

  /// Move constructor
  Vector(Vector&& x) noexcept = default;

  /// Destructor
  ~Vector() = default;

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Get local part of the vector (const version)
  std::shared_ptr<const common::IndexMap> map() const { return _map; }

  /// Get block size
  int bs() const { return _bs; }

  /// Get local part of the vector (const version)
  // const Eigen::Matrix<T, Eigen::Dynamic, 1>& array() const { return _x; }
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> array() const
  {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(_x.data(),
                                                                 _x.size());
  }

  /// Get local part of the vector
  // Eigen::Matrix<T, Eigen::Dynamic, 1>& array() { return _x; }
  // { return _x; }
  std::vector<T>& mutable_array() { return _x; }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  // Data
  // Eigen::Matrix<T, Eigen::Dynamic, 1> _x;
  std::vector<T> _x;
};
} // namespace dolfinx::la
