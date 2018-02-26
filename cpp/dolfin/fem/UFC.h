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

using EigenRowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class UFC
{
public:
  /// Constructor
  UFC(const Form& form);

  /// Destructor
  ~UFC(){};

  /// Update current cell
  void update(const Cell& cell,
              Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs0,
              const ufc::cell& ufc_cell,
              const std::vector<bool>& enabled_coefficients);

  /// Update current cell
  void update(const Cell& cell, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell,
              const std::vector<bool>& enabled_coefficients);

  /// Update current pair of cells for macro element
  void update(const Cell& cell0, const std::vector<double>& coordinate_dofs0,
              const ufc::cell& ufc_cell0, const Cell& cell1,
              const std::vector<double>& coordinate_dofs1,
              const ufc::cell& ufc_cell1,
              const std::vector<bool>& enabled_coefficients);

  /// Update current pair of cells for macro element
  void update(const Cell& cell0,
              Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs0,
              const ufc::cell& ufc_cell0, const Cell& cell1,
              Eigen::Ref<const EigenRowMatrixXd> coordinate_dofs1,
              const ufc::cell& ufc_cell1,
              const std::vector<bool>& enabled_coefficients);

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

  /// Local tensor
  std::vector<double> A;

  /// Local tensor for macro element
  std::vector<double> macro_A;

private:
  // Coefficients (std::vector<double*> is used to interface with
  // UFC)
  std::vector<double> _w;
  std::vector<double*> w_pointer;

  // Coefficients on macro element (std::vector<double*> is used to
  // interface with UFC)
  std::vector<double> _macro_w;
  std::vector<double*> macro_w_pointer;

public:
  /// The form
  const Form& dolfin_form;
};
}
