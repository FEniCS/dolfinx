// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>

namespace dolfinx::common
{

/// This is a singleton class that return IDs that are unique in the
/// lifetime of a program.

class UniqueIdGenerator
{
public:
  UniqueIdGenerator();

  /// Generate a unique ID
  static std::size_t id();

private:
  // Singleton instance
  static UniqueIdGenerator unique_id_generator;

  // Next ID to be returned
  std::size_t _next_id;
};
} // namespace dolfinx::common
