// Copyright (C) 2008-20011 Anders Logg and Garth N. Wells
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
// First added:  2008-09-11
// Last changed: 2013-01-27

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "UFCCell.h"

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    /// Create finite element from UFC finite element (data may be shared)
    FiniteElement(boost::shared_ptr<const ufc::finite_element> element);

    /// Destructor
    virtual ~FiniteElement() {}

    //--- Direct wrappers for ufc::finite_element ---

    /// Return a string identifying the finite element
    std::string signature() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->signature();
    }

    /// Return the cell shape
    ufc::shape cell_shape() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->cell_shape();
    }

    // Return the topological dimension of the cell shape
    std::size_t topological_dimension() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->topological_dimension();
    }

    // Return the geometric dimension of the cell shape
    virtual unsigned int geometric_dimension() const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->geometric_dimension();
    }

    /// Return the dimension of the finite element function space
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
    void evaluate_basis(std::size_t i,
                        double* values,
                        const double* x,
                        const double* vertex_coordinates) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis(i, values, x, vertex_coordinates);
    }

    /// Evaluate all basis functions at given point in cell
    void evaluate_basis_all(double* values,
                            const double* x,
                            const double* vertex_coordinates) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_all(values, x, vertex_coordinates);
    }

    /// Evaluate order n derivatives of basis function i at given point in cell
    void evaluate_basis_derivatives(unsigned int i,
                                    unsigned int n,
                                    double* values,
                                    const double* x,
                                    const double* vertex_coordinates) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives(i, n, values, x, vertex_coordinates);
    }

    /// Evaluate order n derivatives of all basis functions at given point in cell
    void evaluate_basis_derivatives_all(unsigned int n,
                                        double* values,
                                        const double* x,
                                        const double* vertex_coordinates) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives_all(n, values, x, vertex_coordinates);
    }

    /// Evaluate linear functional for dof i on the function f
    double evaluate_dof(std::size_t i,
                        const ufc::function& function,
                        const double* vertex_coordinates,
                        const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      return _ufc_element->evaluate_dof(i, function, vertex_coordinates, c);
    }

    /// Evaluate linear functionals for all dofs on the function f
    void evaluate_dofs(double* values,
                       const ufc::function& f,
                       const double* vertex_coordinates,
                       const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->evaluate_dofs(values, f, vertex_coordinates,c);
    }

    /// Interpolate vertex values from dof values
    void interpolate_vertex_values(double* vertex_values,
                                   double* coefficients,
                                   const ufc::cell& cell) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->interpolate_vertex_values(vertex_values, coefficients, cell);
    }

    /// Map coordinate xhat from reference cell to coordinate x in cell
    void map_from_reference_cell(double* x,
                                 const double* xhat,
                                 const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->map_from_reference_cell(x, xhat, c);
    }

    /// Map from coordinate x in cell to coordinate xhat in reference cell
    void map_to_reference_cell(double* xhat,
                               const double* x,
                               const ufc::cell& c) const
    {
      dolfin_assert(_ufc_element);
      _ufc_element->map_to_reference_cell(xhat, x, c);
    }

    /// Return the number of sub elements (for a mixed element)
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
    boost::shared_ptr<const FiniteElement> create_sub_element(std::size_t i) const
    {
      dolfin_assert(_ufc_element);
      boost::shared_ptr<const ufc::finite_element>  ufc_element(_ufc_element->create_sub_element(i));
      boost::shared_ptr<const FiniteElement> element(new const FiniteElement(ufc_element));
      return element;
    }

    /// Create a new class instance
    boost::shared_ptr<const FiniteElement> create() const
    {
      dolfin_assert(_ufc_element);
      boost::shared_ptr<const ufc::finite_element> ufc_element(_ufc_element->create());
      return boost::shared_ptr<const FiniteElement>(new FiniteElement(ufc_element));
    }

    /// Extract sub finite element for component
    boost::shared_ptr<const FiniteElement> extract_sub_element(const std::vector<std::size_t>& component) const;

  private:

    // Recursively extract sub finite element
    static boost::shared_ptr<const FiniteElement> extract_sub_element(const FiniteElement& finite_element,
                                                                      const std::vector<std::size_t>& component);

    // UFC finite element
    boost::shared_ptr<const ufc::finite_element> _ufc_element;

    // Simple hash of the signature string
    std::size_t _hash;

  };

}
#endif
