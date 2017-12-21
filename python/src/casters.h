// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef _DOLFIN_PYBIND11_CASTERS
#define _DOLFIN_PYBIND11_CASTERS

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/variant.hpp>
#include <boost/optional.hpp>

#include "mpi_casters.h"
#include "petsc_casters.h"

namespace pybind11
{
  namespace detail
  {
    // Caster for boost::optional
    template <typename T>
      struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};

    // Caster for boost::variant (from pybind11 docs)
    template <typename... Ts>
      struct type_caster<boost::variant<Ts...>> : variant_caster<boost::variant<Ts...>> {};

    // Specifies the function used to visit the variant --
    // `apply_visitor` instead of `visit`
    template <>
      struct visit_helper<boost::variant> {
      template <typename... Args>
        static auto call(Args &&...args) -> decltype(boost::apply_visitor(args...)) {
        return boost::apply_visitor(args...);
      }
    };
  }
}

#endif
