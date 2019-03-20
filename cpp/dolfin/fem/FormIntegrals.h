// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <petscsys.h>
#include <vector>

struct ufc_cell_integral;
struct ufc_exterior_facet_integral;
struct ufc_interior_facet_integral;
struct ufc_vertex_integral;
struct ufc_form;

namespace dolfin
{
namespace fem
{

// FIXME: This class would be greatly simplified if all integrals types
// (cell, facet, etc) were the same type.

/// Integrals of a Form, including those defined over cells, interior
/// and exterior facets, and vertices.
class FormIntegrals
{
public:
  /// Type of integral
  enum class Type : std::int8_t
  {
    cell = 0,
    exterior_facet = 1,
    interior_facet = 2,
    vertex = 3
  };

  /// Initialise the FormIntegrals as empty
  FormIntegrals();

  /// Initialise the FormIntegrals from a ufc::form instantiating all
  /// the required integrals
  FormIntegrals(const ufc_form& ufc_form);

  /// Get the function for 'tabulate_tensor' for cell integral i
  /// @param i
  ///    Integral number
  /// @returns std::function
  ///    Function to call for tabulate_tensor on a cell
  const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                           int)>&
  get_tabulate_tensor_fn_cell(int i) const;

  const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                           int)>&
  get_tabulate_tensor_fn_exterior_facet(int i) const;

  /// Set the function for 'tabulate_tensor' for cell integral i
  void set_tabulate_tensor_cell(int i,
                                void (*fn)(PetscScalar*, const PetscScalar*,
                                           const double*, int));

  /// Set the function for 'tabulate_tensor' for exterior facet integral
  /// i
  void set_tabulate_tensor_exterior_facet(int i,
                                          void (*fn)(PetscScalar*,
                                                     const PetscScalar*,
                                                     const double*, int, int));

  /// Number of integrals of given type
  int num_integrals(FormIntegrals::Type t) const;

private:
  // Function pointers to tabulate_tensor functions
  std::vector<
      std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>>
      _tabulate_tensor_cell;
  std::vector<std::function<void(PetscScalar*, const PetscScalar*,
                                 const double*, int, int)>>
      _tabulate_tensor_exterior_facet;
};
} // namespace fem
} // namespace dolfin
