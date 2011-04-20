// Copyright (C) 2008-20011 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2011-04-13

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
    ~FiniteElement() {}

    //--- Direct wrappers for ufc::finite_element ---

    /// Return a string identifying the finite element
    std::string signature() const
    {
      assert(_ufc_element);
      return _ufc_element->signature();
    }

    /// Return the cell shape
    ufc::shape cell_shape() const
    {
      assert(_ufc_element);
      return _ufc_element->cell_shape();
    }

    // Return the topological dimension of the cell shape
    uint topological_dimension() const
    {
      assert(_ufc_element);
      return _ufc_element->topological_dimension();
    }

    // Return the geometric dimension of the cell shape
    virtual unsigned int geometric_dimension() const
    {
      assert(_ufc_element);
      return _ufc_element->geometric_dimension();
    }

    /// Return the dimension of the finite element function space
    uint space_dimension() const
    {
      assert(_ufc_element);
      return _ufc_element->space_dimension();
    }

    /// Return the rank of the value space
    uint value_rank() const
    {
      assert(_ufc_element);
      return _ufc_element->value_rank();
    }

    /// Return the dimension of the value space for axis i
    uint value_dimension(uint i) const
    {
      assert(_ufc_element);
      return _ufc_element->value_dimension(i);
    }

    /// Evaluate basis function i at given point in cell
    void evaluate_basis(uint i, double* values, const double* x,
                        const ufc::cell& cell) const
    {
      assert(_ufc_element);
      _ufc_element->evaluate_basis(i, values, x, cell);
    }

    /// Evaluate all basis functions at given point in cell
    void evaluate_basis_all(double* values,
                            const double* coordinates,
                            const ufc::cell& c) const
    {
      assert(_ufc_element);
      _ufc_element->evaluate_basis_all(values, coordinates, c);
    }

    /// Evaluate order n derivatives of basis function i at given point in cell
    void evaluate_basis_derivatives(unsigned int i,
                                    unsigned int n,
                                    double* values,
                                    const double* x,
                                    const ufc::cell& cell) const
    {
      assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives(i, n, values, x, cell);
    }

    /// Evaluate order n derivatives of all basis functions at given point in cell
    void evaluate_basis_derivatives_all(unsigned int n,
                                        double* values,
                                        const double* coordinates,
                                        const ufc::cell& c) const
    {
      assert(_ufc_element);
      _ufc_element->evaluate_basis_derivatives_all(n, values, coordinates, c);
    }

    /// Evaluate linear functional for dof i on the function f
    double evaluate_dof(uint i,
                        const ufc::function& function,
                        const ufc::cell& cell) const
    {
      assert(_ufc_element);
      return _ufc_element->evaluate_dof(i, function, cell);
    }

    /// Evaluate linear functionals for all dofs on the function f
    void evaluate_dofs(double* values,
                       const ufc::function& f,
                       const ufc::cell& c) const
    {
      assert(_ufc_element);
      _ufc_element->evaluate_dofs(values, f, c);
    }

    /// Interpolate vertex values from dof values
    void interpolate_vertex_values(double* vertex_values,
                                   double* coefficients,
                                   const ufc::cell& cell) const
    {
      assert(_ufc_element);
      _ufc_element->interpolate_vertex_values(vertex_values, coefficients, cell);
    }

    /// Map coordinate xhat from reference cell to coordinate x in cell
    void map_from_reference_cell(double* x,
                                 const double* xhat,
                                 const ufc::cell& c) const
    {
      assert(_ufc_element);
      _ufc_element->map_from_reference_cell(x, xhat, c);
    }

    /// Map from coordinate x in cell to coordinate xhat in reference cell
    void map_to_reference_cell(double* xhat,
                               const double* x,
                               const ufc::cell& c) const
    {
      assert(_ufc_element);
      _ufc_element->map_to_reference_cell(xhat, x, c);
    }

    /// Return the number of sub elements (for a mixed element)
    uint num_sub_elements() const
    {
      assert(_ufc_element);
      return _ufc_element->num_sub_elements();
    }

    //--- DOLFIN-specific extensions of the interface ---

    /// Return simple hash of the signature string
    uint hash() const
    { return _hash; }

    /// Evaluate basis function i at given point in cell
    void evaluate_basis(uint i, double* values, const double* x,
                        const Cell& cell) const
    {
      assert(_ufc_element);
      UFCCell ufc_cell(cell);
      _ufc_element->evaluate_basis(i, values, x, ufc_cell);
    }

    /// Evaluate all basis functions at given point in cell
    void evaluate_basis_all(double* values, const double* coordinates,
			    const Cell& cell) const
    {
      assert(_ufc_element);
      UFCCell ufc_cell(cell);
      _ufc_element->evaluate_basis_all(values, coordinates, ufc_cell);
    }

    /// Create a new finite element for sub element i (for a mixed element)
    boost::shared_ptr<const FiniteElement> create_sub_element(uint i) const
    {
      assert(_ufc_element);
      boost::shared_ptr<const ufc::finite_element>  ufc_element(_ufc_element->create_sub_element(i));
      boost::shared_ptr<const FiniteElement> element(new const FiniteElement(ufc_element));
      return element;
    }

    /// Create a new class instance
    boost::shared_ptr<const FiniteElement> create() const
    {
      assert(_ufc_element);
      boost::shared_ptr<const ufc::finite_element> ufc_element(_ufc_element->create());
      return boost::shared_ptr<const FiniteElement>(new FiniteElement(ufc_element));
    }

    /// Extract sub finite element for component
    boost::shared_ptr<const FiniteElement> extract_sub_element(const std::vector<uint>& component) const;

  private:

    // Recursively extract sub finite element
    static boost::shared_ptr<const FiniteElement> extract_sub_element(const FiniteElement& finite_element,
                                                                      const std::vector<uint>& component);

    // UFC finite element
    boost::shared_ptr<const ufc::finite_element> _ufc_element;

    // Simple hash of the signature string
    uint _hash;

  };

}
#endif
