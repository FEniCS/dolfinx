// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2009-12-11

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <boost/shared_ptr.hpp>
#include <vector>
#include <dolfin/common/types.h>
#include "UFC.h"
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

    std::string signature() const
    { return _ufc_element->signature(); }

    /// Return simple hash of the signature string
    uint hash() const
    { return _hash; }

    uint value_rank() const
    { return _ufc_element->value_rank(); }

    uint value_dimension(uint i) const
    { return _ufc_element->value_dimension(i); }

    uint num_sub_elements() const
    { return _ufc_element->num_sub_elements(); }

    uint space_dimension() const
    { return _ufc_element->space_dimension(); }

    void interpolate_vertex_values(double* vertex_values, double* coefficients,
                                   const ufc::cell& cell) const
    { _ufc_element->interpolate_vertex_values(vertex_values, coefficients, cell); }

    void evaluate_basis(uint i, double* values, const double* x,
                        const ufc::cell& cell) const
    { _ufc_element->evaluate_basis(i, values, x, cell); }

    void evaluate_basis(uint i, double* values, const double* x,
                        const Cell& cell) const
    { UFCCell ufc_cell(cell); _ufc_element->evaluate_basis(i, values, x, ufc_cell); }

    void evaluate_basis_derivatives(unsigned int i,
                                    unsigned int n,
                                    double* values,
                                    const double* x,
                                    const ufc::cell& cell) const
    { _ufc_element->evaluate_basis_derivatives(i, n, values, x, cell); }

    double evaluate_dof(uint i, const ufc::function& function,
                        const ufc::cell& cell) const
    {
      assert(_ufc_element);
      return _ufc_element->evaluate_dof(i, function, cell);
    }

    /// Create sub element
    boost::shared_ptr<const FiniteElement> create_sub_element(uint i) const
    {
      boost::shared_ptr<const ufc::finite_element>  ufc_element(_ufc_element->create_sub_element(i));
      boost::shared_ptr<const FiniteElement> element(new const FiniteElement(ufc_element));
      return element;
    }

    /// Return ufc::finite_element
    boost::shared_ptr<const ufc::finite_element> ufc_element() const
    { return _ufc_element; }

    /// Extract sub finite element for component
    boost::shared_ptr<const FiniteElement> extract_sub_element(const std::vector<uint>& component) const;

  private:

    // Friends
    friend class AdaptiveObjects;

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
