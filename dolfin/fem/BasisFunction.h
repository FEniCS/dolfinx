// Copyright (C) 2013 Anders Logg
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
//
// First added:  2009-01-01
// Last changed: 2013-03-04

#ifndef __BASIS_FUNCTION_H
#define __BASIS_FUNCTION_H

#include <memory>
#include <vector>
#include <ufc.h>
#include <dolfin/fem/FiniteElement.h>

namespace dolfin
{

  /// Represention of a finite element basis function.

  /// It can be used for computation of basis function values
  /// and derivatives.
  ///
  /// Evaluation of basis functions is also possible through the use
  /// of the functions ``evaluate_basis`` and
  /// ``evaluate_basis_derivatives`` available in the _FiniteElement_
  /// class. The BasisFunction class relies on these functions for
  /// evaluation but also implements the ufc::function interface which
  /// allows evaluate_dof to be evaluated for a basis function (on a
  /// possibly different element).

  class BasisFunction : public ufc::function
  {
  public:

    /// Create basis function with given index on element on given cell
    ///
    /// @param    index (std::size_t)
    ///         The index of the basis function.
    /// @param    element (_FiniteElement_)
    ///         The element to create basis function on.
    /// @param  coordinate_dofs (std::vector<double>&)
    ///         The coordinate dofs of the cell
    BasisFunction(std::size_t index,
                  std::shared_ptr<const FiniteElement> element,
                  const std::vector<double>& coordinate_dofs)
      : _index(index), _element(element), _coordinate_dofs(coordinate_dofs) {}

    /// Destructor
    ~BasisFunction() {}

    /// Update the basis function index
    ///
    /// @param    index (std::size_t)
    ///         The index of the basis function.
    void update_index(std::size_t index)
    { _index = index; }

    /// Evaluate basis function at given point
    ///
    /// @param    values (double)
    ///         The values of the function at the point.
    /// @param    x (double)
    ///         The coordinates of the point.
    void eval(double* values, const double* x) const
    {
      // Note: assuming cell_orientation = 0
      dolfin_assert(_element);
      _element->evaluate_basis(_index, values, x, _coordinate_dofs.data(), 0);
    }

    /// Evaluate all order n derivatives at given point
    ///
    /// @param    values (double)
    ///         The values of derivatives at the point.
    /// @param    x (double)
    ///         The coordinates of the point.
    /// @param    n (std::size_t)
    ///         The order of derivation.
    void eval_derivatives(double* values, const double* x, std::size_t n) const
    {
      // Note: assuming cell_orientation = 0
      dolfin_assert(_element);
      _element->evaluate_basis_derivatives(_index, n, values, x,
                                           _coordinate_dofs.data(), 0);
    }

    //--- Implementation of ufc::function interface ---

    /// Evaluate function at given point in cell
    ///
    /// @param    values (double)
    ///         The values of the function at the point..
    /// @param    coordinates (double)
    ///         The coordinates of the point.
    /// @param    cell (ufc::cell)
    ///         The cell.
    void evaluate(double* values, const double* coordinates,
                  const ufc::cell& cell) const
    { eval(values, coordinates); }

  private:

    // The index
    std::size_t _index;

    // The finite element
    std::shared_ptr<const FiniteElement> _element;

    // Cell coordinates
    const std::vector<double> _coordinate_dofs;

  };

}

#endif
