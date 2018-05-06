// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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
  enum class Type
  {
    cell,
    exterior_facet,
    interior_facet,
    vertex
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
  const std::function<void(double*, const double* const*, const double*, int)>&
  cell_tabulate_tensor(int i) const;

  const bool* cell_enabled_coefficients(int i) const;

  /// Set the function for 'tabulate_tensor' for cell integral i
  void set_cell_tabulate_tensor(int i, void (*fn)(double*, const double* const*,
                                                  const double*, int));

  /// Number of integrals of given type
  int count(FormIntegrals::Type t) const;

  /// Number of cell integrals
  int num_cell_integrals() const;

  /// Default exterior facet integral
  std::shared_ptr<const ufc_exterior_facet_integral>
  exterior_facet_integral() const;

  /// Exterior facet integral for domain i
  std::shared_ptr<const ufc_exterior_facet_integral>
  exterior_facet_integral(unsigned int i) const;

  /// Number of exterior facet integrals
  int num_exterior_facet_integrals() const;

  /// Default interior facet integral
  std::shared_ptr<const ufc_interior_facet_integral>
  interior_facet_integral() const;

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc_interior_facet_integral>
  interior_facet_integral(unsigned int i) const;

  /// Number of interior facet integrals
  int num_interior_facet_integrals() const;

  /// Default interior facet integral
  std::shared_ptr<const ufc_vertex_integral> vertex_integral() const;

  /// Interior facet integral for domain i
  std::shared_ptr<const ufc_vertex_integral>
  vertex_integral(unsigned int i) const;

  /// Number of vertex integrals
  int num_vertex_integrals() const;

private:
  // Cell integrals
  std::vector<std::shared_ptr<ufc_cell_integral>> _cell_integrals;

  // Function pointers to cell tabulate_tensor functions
  std::vector<
      std::function<void(double*, const double* const*, const double*, int)>>
      _cell_tabulate_tensor;

  // Storage for enabled coefficients, to match the functions
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      _enabled_coefficients;

  // Exterior facet integrals
  std::vector<std::shared_ptr<ufc_exterior_facet_integral>>
      _exterior_facet_integrals;

  // Interior facet integrals
  std::vector<std::shared_ptr<ufc_interior_facet_integral>>
      _interior_facet_integrals;

  // Vertex integrals
  std::vector<std::shared_ptr<ufc_vertex_integral>> _vertex_integrals;
};
} // namespace fem
} // namespace dolfin
