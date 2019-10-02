// Copyright (C) 2007-2008 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{
namespace mesh
{

class Mesh;

/// This class implements the ordering of simples mesh entities
/// according to the UFC specification.

class Ordering
{
public:
  /// Order mesh
  /// @param[in,out] mesh The mesh to be re-ordered
  static void order_simplex(Mesh& mesh);

  /// Check if mesh is ordered
  /// @param[in] mesh The mesh to checked
  /// @return True if mesh is ordered, otherwise false
  static bool is_ordered_simplex(const Mesh& mesh);
};

} // namespace mesh
} // namespace dolfin
