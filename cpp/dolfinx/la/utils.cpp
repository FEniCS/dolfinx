// Copyright(C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Vector.h"

template <typename T>
void dolfinx::la::scatter_fwd(dolfinx::la::Vector<T>& v)
{
  xtl::span<const T> xlocal(v.array().data(), v.map()->size_local() * v.bs());
  xtl::span<T> xremote(v.mutable_array().data()
                           + v.map()->size_local() * v.bs(),
                       v.map()->num_ghosts() * v.bs());
  v.map()->scatter_fwd(xlocal, xremote, v.bs());
}

template <typename T>
void dolfinx::la::scatter_rev(dolfinx::la::Vector<T>& v,
                              dolfinx::common::IndexMap::Mode op)
{
  xtl::span<T> xlocal(v.mutable_array().data(), v.map()->size_local() * v.bs());
  xtl::span<const T> xremote(v.array().data() + v.map()->size_local() * v.bs(),
                             v.map()->num_ghosts() * v.bs());
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

//-----------------------------------------------------------------------------
template <typename T>
T dolfinx::la::norm(const dolfinx::la::Vector<T>& v)
{
  const std::vector<T>& arr = v.array();
  const std::int32_t size_local = v.map()->size_local();

  double result = std::transform_reduce(arr.data(), arr.data() + size_local,
                                        0.0, std::plus<double>(),
                                        [](T val) { return std::norm(val); });

  double global_result;
  MPI_Allreduce(&result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                v.map()->comm());

  return std::sqrt(global_result);
}

/// @cond
// Instantiate
template double dolfinx::la::norm(const dolfinx::la::Vector<double>&);
template float dolfinx::la::norm(const dolfinx::la::Vector<float>&);
template std::complex<double>
dolfinx::la::norm(const dolfinx::la::Vector<std::complex<double>>&);
template std::complex<float>
dolfinx::la::norm(const dolfinx::la::Vector<std::complex<float>>&);
/// @endcond

template <typename T>
T dolfinx::la::max(const dolfinx::la::Vector<T>& v)
{
  const std::vector<T>& arr = v.array();
  const std::int32_t size_local = v.map()->size_local();

  T result = std::reduce(arr.data(), arr.data() + size_local, 0.0,
                         [](T a, T b) { return std::max(a, b); });
  return result;
}

/// @cond
template <>
std::complex<double>
dolfinx::la::max(const dolfinx::la::Vector<std::complex<double>>&)
{
  throw std::runtime_error("Cannot compute max of a complex vector");
}
template <>
std::complex<float>
dolfinx::la::max(const dolfinx::la::Vector<std::complex<float>>&)
{
  throw std::runtime_error("Cannot compute max of a complex vector");
}
template double dolfinx::la::max(const dolfinx::la::Vector<double>&);
template float dolfinx::la::max(const dolfinx::la::Vector<float>&);
/// @endcond