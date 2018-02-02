// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <ufc.h>
#include <vector>

namespace dolfin
{

class Cell;
class FiniteElement;
class Form;
class FunctionSpace;
class GenericFunction;
class Mesh;

/// This class is a simple data structure that holds data used
/// during assembly of a given UFC form. Data is created for each
/// primary argument, that is, v_j for j < r. In addition, nodal
/// basis expansion coefficients and a finite element are created
/// for each coefficient function.

class UFC
{
public:
  /// Constructor
  UFC(const Form& form);

  /// Copy constructor
  UFC(const UFC& ufc) = delete;

  /// Destructor
  ~UFC() = default;

  /// Update current cell
  void update(const Cell& cell, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell,
              const std::vector<bool>& enabled_coefficients);

  /// Update current cell
  void update(const Cell& cell,
              Eigen::Ref<const Eigen::MatrixXd> coordinate_dofs0,
              const ufc::cell& ufc_cell,
              const std::vector<bool>& enabled_coefficients);

  /// Update current pair of cells for macro element
  void update(const Cell& cell0, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell0, const Cell& cell1,
              const std::vector<double>& coordinate_dofs1,
              const ufc::cell& ufc_cell1,
              const std::vector<bool>& enabled_coefficients);

  /// Update current cell (TODO: Remove this when
  /// PointIntegralSolver supports the version with
  /// enabled_coefficients)
  void update(const Cell& cell, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell);

  /// Update current pair of cells for macro element (TODO: Remove
  /// this when PointIntegralSolver supports the version with
  /// enabled_coefficients)
  void update(const Cell& cell0, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell0, const Cell& cell1,
              const std::vector<double>& coordinate_dofs1,
              const ufc::cell& ufc_cell1);

  /// Pointer to coefficient data. Used to support UFC interface.
  const double* const* w() const { return w_pointer.data(); }

  /// Pointer to coefficient data. Used to support UFC
  /// interface. None const version
  double** w() { return w_pointer.data(); }

  /// Pointer to macro element coefficient data. Used to support UFC
  /// interface.
  const double* const* macro_w() const { return macro_w_pointer.data(); }

  /// Pointer to macro element coefficient data. Used to support UFC
  /// interface. Non-const version.
  double** macro_w() { return macro_w_pointer.data(); }

private:
  // Finite elements for coefficients
  std::vector<FiniteElement> coefficient_elements;

  // Cell integrals (access through get_cell_integral to get proper
  // fallback to default)
  std::vector<std::shared_ptr<ufc::cell_integral>> cell_integrals;

  // Exterior facet integrals (access through
  // get_exterior_facet_integral to get proper fallback to default)
  std::vector<std::shared_ptr<ufc::exterior_facet_integral>>
      exterior_facet_integrals;

  // Interior facet integrals (access through
  // get_interior_facet_integral to get proper fallback to default)
  std::vector<std::shared_ptr<ufc::interior_facet_integral>>
      interior_facet_integrals;

  // Point integrals (access through get_vertex_integral to get
  // proper fallback to default)
  std::vector<std::shared_ptr<ufc::vertex_integral>> vertex_integrals;

public:
  // Default cell integral
  std::shared_ptr<ufc::cell_integral> default_cell_integral;

  // Default exterior facet integral
  std::shared_ptr<ufc::exterior_facet_integral> default_exterior_facet_integral;

  // Default interior facet integral
  std::shared_ptr<ufc::interior_facet_integral> default_interior_facet_integral;

  // Default point integral
  std::shared_ptr<ufc::vertex_integral> default_vertex_integral;

  /// Get cell integral over a given domain, falling back to the
  /// default if necessary
  ufc::cell_integral* get_cell_integral(std::size_t domain)
  {
    if (domain < form.max_cell_subdomain_id())
    {
      ufc::cell_integral* integral = cell_integrals[domain].get();
      if (integral)
        return integral;
    }
    return default_cell_integral.get();
  }

  /// Get exterior facet integral over a given domain, falling back
  /// to the default if necessary
  ufc::exterior_facet_integral* get_exterior_facet_integral(std::size_t domain)
  {
    if (domain < form.max_exterior_facet_subdomain_id())
    {
      ufc::exterior_facet_integral* integral
          = exterior_facet_integrals[domain].get();
      if (integral)
        return integral;
    }
    return default_exterior_facet_integral.get();
  }

  /// Get interior facet integral over a given domain, falling back
  /// to the default if necessary
  ufc::interior_facet_integral* get_interior_facet_integral(std::size_t domain)
  {
    if (domain < form.max_interior_facet_subdomain_id())
    {
      ufc::interior_facet_integral* integral
          = interior_facet_integrals[domain].get();
      if (integral)
        return integral;
    }
    return default_interior_facet_integral.get();
  }

  /// Get point integral over a given domain, falling back to the
  /// default if necessary
  ufc::vertex_integral* get_vertex_integral(std::size_t domain)
  {
    if (domain < form.max_vertex_subdomain_id())
    {
      ufc::vertex_integral* integral = vertex_integrals[domain].get();
      if (integral)
        return integral;
    }
    return default_vertex_integral.get();
  }

  /// Form
  const ufc::form& form;

  /// Local tensor
  std::vector<double> A;

  /// Local tensor
  std::vector<double> A_facet;

  /// Local tensor for macro element
  std::vector<double> macro_A;

private:
  // Coefficients (std::vector<double*> is used to interface with
  // UFC)
  std::vector<std::vector<double>> _w;
  std::vector<double*> w_pointer;

  // Coefficients on macro element (std::vector<double*> is used to
  // interface with UFC)
  std::vector<std::vector<double>> _macro_w;
  std::vector<double*> macro_w_pointer;

  // Coefficient functions
  const std::vector<std::shared_ptr<const GenericFunction>> coefficients;

public:
  /// The form
  const Form& dolfin_form;
};
}
