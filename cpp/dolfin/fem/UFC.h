// Copyright (C) 2007-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
}

namespace fem
{
class FiniteElement;
class Form;

/// This class is a simple data structure that holds data used during
/// assembly of a given UFC form. Data is created for each primary
/// argument, that is, v_j for j < r. In addition, nodal basis expansion
/// coefficients and a finite element are created for each coefficient
/// function.

class UFC
{
public:
  /// Constructor
  UFC(const Form& form);

  /// Destructor
  ~UFC(){};

  /// Update current cell
  void
  update(const mesh::Cell& cell,
         const Eigen::Ref<const dolfin::EigenRowArrayXXd>& coordinate_dofs0,
         const bool* enabled_coefficients);

  /// Update current pair of cells for macro element
  void
  update(const mesh::Cell& cell0,
         const Eigen::Ref<const dolfin::EigenRowArrayXXd>& coordinate_dofs0,
         const mesh::Cell& cell1,
         const Eigen::Ref<const dolfin::EigenRowArrayXXd>& coordinate_dofs1,
         const bool* enabled_coefficients);

  /// Pointer to coefficient data. Used to support UFC interface.
  const PetscScalar* const* w() const { return w_pointer.data(); }

  /// Pointer to coefficient data. Used to support UFC interface. None
  /// const version
  PetscScalar** w() { return w_pointer.data(); }

  /// Pointer to macro element coefficient data. Used to support UFC
  /// interface.
  const PetscScalar* const* macro_w() const { return macro_w_pointer.data(); }

  /// Pointer to macro element coefficient data. Used to support UFC
  /// interface. Non-const version.
  PetscScalar** macro_w() { return macro_w_pointer.data(); }

  /// Local tensor
  std::vector<PetscScalar> A;

  /// Local tensor for macro element
  std::vector<PetscScalar> macro_A;

private:
  // Coefficients (std::vector<PetscScalar*> is used to interface with UFC)
  std::vector<PetscScalar> _w;
  std::vector<PetscScalar*> w_pointer;

  // Coefficients on macro element (std::vector<PetscScalar*> is used to
  // interface with UFC)
  std::vector<PetscScalar> _macro_w;
  std::vector<PetscScalar*> macro_w_pointer;

public:
  /// The form
  const Form& dolfin_form;
};
} // namespace fem
} // namespace dolfin
