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

template <typename T>
void dolfinx::la::scatter_fwd(dolfinx::la::Vector<T>& v)
{
  tcb::span<const T> xlocal(v.array().data(), v.map()->size_local());
  tcb::span<T> xremote(v.mutable_array().data() + v.map()->size_local(),
                       v.map()->num_ghosts());
  v.map()->scatter_fwd(xlocal, xremote, v.bs());
}

template <typename T>
void dolfinx::la::scatter_rev(dolfinx::la::Vector<T>& v,
                              dolfinx::common::IndexMap::Mode op)
{
  tcb::span<T> xlocal(v.mutable_array().data(), v.map()->size_local());
  tcb::span<const T> xremote(v.array().data() + v.map()->size_local(),
                             v.map()->num_ghosts());
  v.map()->scatter_rev(xlocal, xremote, v.bs(), op);
}

// \cond turn off doxygen
template void dolfinx::la::scatter_fwd<double>(dolfinx::la::Vector<double>& v);
template void dolfinx::la::scatter_fwd<std::complex<double>>(
    dolfinx::la::Vector<std::complex<double>>& v);

template void
dolfinx::la::scatter_rev<double>(dolfinx::la::Vector<double>& v,
                                 dolfinx::common::IndexMap::Mode op);
template void dolfinx::la::scatter_rev<std::complex<double>>(
    dolfinx::la::Vector<std::complex<double>>& v,
    dolfinx::common::IndexMap::Mode op);
// \endcond