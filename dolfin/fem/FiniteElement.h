// Copyright (C) 2008-2013 Anders Logg and Garth N. Wells
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

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <memory>
#include <vector>
#include <ufc.h>
#include <boost/multi_array.hpp>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class Cell;

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    /// Create finite element from UFC finite element (data may be shared)
    /// @param element
    ///  UFC finite element
    FiniteElement(std::shared_ptr<const ufc::finite_element> element);

    /// Destructor
    virtual ~FiniteElement() {}

    //--- Direct wrappers for ufc::finite_element ---

    /// Return a string identifying the finite element
    /// @return std::string
    std::string signature() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->signature();
    }

    /// Return the cell shape
    /// @return ufc::shape
    ufc::shape cell_shape() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->cell_shape();
    }

    /// Return the topological dimension of the cell shape
    /// @return std::size_t
    std::size_t topological_dimension() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->topological_dimension();
    }

    /// Return the geometric dimension of the cell shape
    /// @return unsigned int
    virtual unsigned int geometric_dimension() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->geometric_dimension();
    }

    /// Return the dimension of the finite element function space
    /// @return std::size_t
    std::size_t space_dimension() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->space_dimension();
    }

    /// Return the rank of the value space
    std::size_t value_rank() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->value_rank();
    }

    /// Return the dimension of the value space for axis i
    std::size_t value_dimension(std::size_t i) const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->value_dimension(i);
    }

    /// Evaluate basis function i at given point in cell
    void evaluate_basis(std::size_t i, double* values, const double* x,
                        const double* coordinate_dofs,
                        int cell_orientation) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis(i, values, x, coordinate_dofs,
                                   cell_orientation);
    }

    /// Evaluate all basis functions at given point in cell
    void evaluate_basis_all(double* values,
                            const double* x,
                            const double* coordinate_dofs,
                            int cell_orientation) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_all(values, x, coordinate_dofs,
                                       cell_orientation);
    }

    /// Evaluate order n derivatives of basis function i at given point in cell
    void evaluate_basis_derivatives(unsigned int i,
                                    unsigned int n,
                                    double* values,
                                    const double* x,
                                    const double* coordinate_dofs,
                                    int cell_orientation) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives(i, n, values, x,
                                               coordinate_dofs,
                                               cell_orientation);
    }

    /// Evaluate order n derivatives of all basis functions at given
    /// point in cell
    void evaluate_basis_derivatives_all(unsigned int n,
                                        double* values,
                                        const double* x,
                                        const double* coordinate_dofs,
                                        int cell_orientation) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives_all(n, values, x,
                                                   coordinate_dofs,
                                                   cell_orientation);
    }

    /// Evaluate linear functional for dof i on the function f
    double evaluate_dof(std::size_t i,
                        const ufc::function& function,
                        const double* coordinate_dofs,
                        int cell_orientation,
                        const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->evaluate_dof(i, function, coordinate_dofs,
                                        cell_orientation, c);
    }

    /// Evaluate linear functionals for all dofs on the function f
    void evaluate_dofs(double* values,
                       const ufc::function& f,
                       const double* coordinate_dofs,
                       int cell_orientation,
                       const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_dofs(values, f, coordinate_dofs,
                                  cell_orientation, c);
    }

    /// Interpolate vertex values from dof values
    /// @param vertex_values
    /// @param coefficients
    /// @param coordinate_dofs
    /// @param cell_orientation
    /// @param cell
    void interpolate_vertex_values(double* vertex_values,
                                   double* coefficients,
                                   const double* coordinate_dofs,
                                   int cell_orientation,
                                   const ufc::cell& cell) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->interpolate_vertex_values(vertex_values, coefficients,
                                              coordinate_dofs,
                                              cell_orientation, cell);
    }

    /// Tabulate the coordinates of all dofs on an element
    ///
    /// @param[in,out]    coordinates (boost::multi_array<double, 2>)
    ///         The coordinates of all dofs on a cell.
    /// @param[in]    coordinate_dofs (std::vector<double>)
    ///         The cell coordinates
    /// @param[in]    cell (Cell)
    ///         The cell.
    void tabulate_dof_coordinates(boost::multi_array<double, 2>& coordinates,
                                  const std::vector<double>& coordinate_dofs,
                                  const Cell& cell) const;

    /// Return the number of sub elements (for a mixed element)
    /// @return std::size_t
    ///   number of sub-elements
    std::size_t num_sub_elements() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->num_sub_elements();
    }

    //--- DOLFIN-specific extensions of the interface ---

    /// Return simple hash of the signature string
    std::size_t hash() const
    { return _hash; }

    /// Create a new finite element for sub element i (for a mixed element)
    std::shared_ptr<const FiniteElement>
      create_sub_element(std::size_t i) const
    {
      dolfin_assert(_ufc_element);
      std::shared_ptr<const ufc::finite_element>
        ufc_element(_ufc_element->create_sub_element(i));
      std::shared_ptr<const FiniteElement>
        element(new const FiniteElement(ufc_element));
      return element;
    }

    /// Create a new class instance
    std::shared_ptr<const FiniteElement> create() const
    {
      dolfin_assert(_ufc_element);
      std::shared_ptr<const ufc::finite_element>
        ufc_element(_ufc_element->create());
      return std::shared_ptr<const FiniteElement>(new FiniteElement(ufc_element));
    }

    /// Extract sub finite element for component
    std::shared_ptr<const FiniteElement>
      extract_sub_element(const std::vector<std::size_t>& component) const;

    /// Return underlying UFC element. Intended for libray usage only
    /// and may change.
    std::shared_ptr<const ufc::finite_element> ufc_element() const
    { return _ufc_element; }

  private:

    // UFC finite element
    std::shared_ptr<const ufc::finite_element> _ufc_element;

    // Recursively extract sub finite element
    static std::shared_ptr<const FiniteElement>
      extract_sub_element(const FiniteElement& finite_element,
                          const std::vector<std::size_t>& component);

    // Simple hash of the signature string
    std::size_t _hash;

  };

}
#endif
