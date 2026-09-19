// Copyright (C) 2013-2026 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Expression.h"
#include "Function.h"
#include "FunctionSpace.h"
#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/EntityMap.h>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <ufcx.h>
#include <utility>
#include <vector>

/// @file expression_factory.h
/// @brief Factories for finite element expressions from UFCx input.

namespace dolfinx::fem
{
/// @brief Create Expression from UFC
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients,
    const std::vector<std::shared_ptr<const Constant<T>>>& constants,
    const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
        entity_maps,
    std::shared_ptr<const FunctionSpace<U>> argument_space = nullptr)
{
  if (e.rank > 0 and !argument_space)
  {
    throw std::invalid_argument("Expression has Argument but no Argument "
                                "function space was provided.");
  }

  std::vector<U> X(e.points, e.points + e.num_points * e.entity_dimension);
  std::array<std::size_t, 2> Xshape
      = {static_cast<std::size_t>(e.num_points),
         static_cast<std::size_t>(e.entity_dimension)};
  std::vector<std::size_t> value_shape(e.value_shape,
                                       e.value_shape + e.num_components);

  static_assert(std::is_same_v<U, scalar_value_t<T>>,
                "UFCx kernels require geometry type U == scalar_value_t<T>.");

  using kptr_t = void (*)(T*, const T*, const T*, const U*, const int*,
                          const std::uint8_t*, void*);
  std::function<void(T*, const T*, const T*, const U*, const int*,
                     const std::uint8_t*, void*)>
      tabulate_tensor = nullptr;
  if constexpr (std::is_same_v<T, float>)
    tabulate_tensor = reinterpret_cast<kptr_t>(e.tabulate_tensor_float32);
  else if constexpr (std::is_same_v<T, double>)
    tabulate_tensor = reinterpret_cast<kptr_t>(e.tabulate_tensor_float64);
#ifndef DOLFINX_NO_STDC_COMPLEX_KERNELS
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    tabulate_tensor = reinterpret_cast<kptr_t>(e.tabulate_tensor_complex64);
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    tabulate_tensor = reinterpret_cast<kptr_t>(e.tabulate_tensor_complex128);
#endif // DOLFINX_NO_STDC_COMPLEX_KERNELS
  else
    throw std::invalid_argument("Type not supported.");

  assert(tabulate_tensor);
  std::uint64_t e_hash = e.coordinate_element_hash;
  return Expression(coefficients, constants, std::span<const U>(X), Xshape,
                    tabulate_tensor, value_shape, entity_maps, e_hash,
                    argument_space);
}

/// @brief Create Expression from UFC input (with named coefficients and
/// constants).
template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
Expression<T, U> create_expression(
    const ufcx_expression& e,
    const std::map<std::string, std::shared_ptr<const Function<T, U>>>&
        coefficients,
    const std::map<std::string, std::shared_ptr<const Constant<T>>>& constants,
    const std::vector<std::reference_wrapper<const mesh::EntityMap>>&
        entity_maps,
    std::shared_ptr<const FunctionSpace<U>> argument_space = nullptr)
{
  // Place coefficients in appropriate order
  std::vector<std::shared_ptr<const Function<T, U>>> coeff_map;
  std::vector<std::string> coefficient_names;
  coefficient_names.reserve(e.num_coefficients);
  for (int i = 0; i < e.num_coefficients; ++i)
    coefficient_names.push_back(e.coefficient_names[i]);

  for (const std::string& name : coefficient_names)
  {
    if (auto it = coefficients.find(name); it != coefficients.end())
      coeff_map.push_back(it->second);
    else
    {
      throw std::runtime_error(
          std::format("Expression coefficient \"{}\" not provided.", name));
    }
  }

  // Place constants in appropriate order
  std::vector<std::shared_ptr<const Constant<T>>> const_map;
  std::vector<std::string> constant_names;
  constant_names.reserve(e.num_constants);
  for (int i = 0; i < e.num_constants; ++i)
    constant_names.push_back(e.constant_names[i]);

  for (const std::string& name : constant_names)
  {
    if (auto it = constants.find(name); it != constants.end())
      const_map.push_back(it->second);
    else
    {
      throw std::runtime_error(
          std::format("Expression constant \"{}\" not provided.", name));
    }
  }

  return create_expression(e, coeff_map, const_map, entity_maps,
                           argument_space);
}
} // namespace dolfinx::fem
