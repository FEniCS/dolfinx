// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "mpi_casters.h"
#include "petsc_casters.h"
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11
{
namespace detail
{
// Caster for boost::optional
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>>
{
};

// Caster for boost::variant (from pybind11 docs)
template <typename... Ts>
struct type_caster<boost::variant<Ts...>>
    : variant_caster<boost::variant<Ts...>>
{
};

// Specifies the function used to visit the variant --
// `apply_visitor` instead of `visit`
template <>
struct visit_helper<boost::variant>
{
  template <typename... Args>
  static auto call(Args&&... args) -> decltype(boost::apply_visitor(args...))
  {
    return boost::apply_visitor(args...);
  }
};
}
}


