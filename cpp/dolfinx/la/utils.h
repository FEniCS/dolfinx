// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/Vector.h>
#pragma once

namespace dolfinx::la
{

/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

/// Scatter la::Vector local data to ghost values.
/// @param[in, out] v la::Vector to update
template <typename T>
void scatter_fwd(Vector<T>& v);

/// Scatter la::Vector ghost data to owner. This process will result in multiple
/// incoming values, which can be summed or inserted into the local vector.
/// @param[in, out] v la::Vector to update
/// @param op IndexMap operation (add or insert)
template <typename T>
void scatter_rev(Vector<T>& v, dolfinx::common::IndexMap::Mode op);

} // namespace dolfinx::la
