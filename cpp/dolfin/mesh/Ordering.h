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
  static void order_simplex(Mesh& mesh);

  /// Check if mesh is ordered
  static bool ordered_simplex(const Mesh& mesh);
};

} // namespace mesh
} // namespace dolfin
