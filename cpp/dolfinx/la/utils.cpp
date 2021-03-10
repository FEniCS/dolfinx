// Copyright(C) 2021 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Vector.h"

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
