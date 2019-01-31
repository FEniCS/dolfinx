// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <functional>
#include <memory>
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

  /// Initialise the FormIntegrals from a ufc::form instantiating all
  /// the required integrals
  FormIntegrals(const ufc_form& ufc_form);

  /// Initialise the FormIntegrals as empty
  FormIntegrals() {}

  /// Default cell integral
  std::shared_ptr<const ufc_cell_integral> cell_integral() const;

  /// Cell integral for domain i
  std::shared_ptr<const ufc_cell_integral> cell_integral(unsigned int i) const;

  /// Get the function for 'tabulate_tensor' for cell integral i
  /// @param i
  ///    Integral number
  /// @returns std::function
  ///    Function to call for tabulate_tensor on a cell
  const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                           int)>&
  tabulate_tensor_cell(int i) const;

  const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                           int)>&
  tabulate_tensor_exterior_facet(int i) const;

  /// Get the enabled coefficients on cell integral i
  /// @param i
  ///    Integral number
  /// @returns bool*
  ///    Pointer to list of enabled coefficients for this integral
  const bool* enabled_coefficients_cell(int i) const;

  const bool* enabled_coefficients_exterior_facet(int i) const;

  /// Set the function for 'tabulate_tensor' for cell integral i
  void set_tabulate_tensor_cell(int i,
                                void (*fn)(PetscScalar*, const PetscScalar*,
                                           const double*, int));

  void set_tabulate_tensor_exterior_facet(int i,
                                          void (*fn)(PetscScalar*,
                                                     const PetscScalar*,
                                                     const double*, int, int));

  /// Number of integrals of given type
  int num_integrals(FormIntegrals::Type t) const;

  /// Default exterior facet integral
  std::shared_ptr<const ufc_exterior_facet_integral>
  exterior_facet_integral() const;

  /// Exterior facet integral for domain i
  std::shared_ptr<const ufc_exterior_facet_integral>
  exterior_facet_integral(unsigned int i) const;

  /// Default interior facet integral
  std::shared_ptr<const ufc_interior_facet_integral>
  interior_facet_integral() const;

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc_interior_facet_integral>
  interior_facet_integral(unsigned int i) const;
  /// Default interior facet integral
  std::shared_ptr<const ufc_vertex_integral> vertex_integral() const;

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc_vertex_integral>
  vertex_integral(unsigned int i) const;

private:
  // Integrals
  std::vector<std::shared_ptr<ufc_cell_integral>> _integrals_cell;
  std::vector<std::shared_ptr<ufc_exterior_facet_integral>>
      _integrals_exterior_facet;

  // Function pointers to cell tabulate_tensor functions
  std::vector<
      std::function<void(PetscScalar*, const PetscScalar*, const double*, int)>>
      _tabulate_tensor_cell;

  std::vector<std::function<void(PetscScalar*, const PetscScalar*,
                                 const double*, int, int)>>
      _tabulate_tensor_exterior_facet;

  // Storage for enabled coefficients, to match the functions
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _enabled_coefficients_cell, _enabled_coefficients_exterior_facet;

  // Interior facet integrals
  std::vector<std::shared_ptr<ufc_interior_facet_integral>>
      _interior_facet_integrals;

  // Vertex integrals
  std::vector<std::shared_ptr<ufc_vertex_integral>> _vertex_integrals;
};
} // namespace fem
} // namespace dolfin
