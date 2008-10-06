// Copyright (C) 2008 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-10-06

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <tr1/memory>
#include <ufc.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <dolfin/elements/ElementLibrary.h>

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    /// Create FiniteElement. Object owns ufc::finite_element
    FiniteElement(ufc::finite_element* element) : element(element) 
    { /* Do nothing */ }

    /// Create FiniteElement. Object may share ufc::finite_element
    FiniteElement(std::tr1::shared_ptr<ufc::finite_element> element) : element(element)
    { /* Do nothing */ }

    /// Create FiniteElement from a signature
    FiniteElement(std::string signature) : element(ElementLibrary::create_finite_element(signature))
    { /* Do nothing */ }

    std::string signature() const
    { return element->signature(); }

    uint value_rank() const
    { return element->value_rank(); }
   
    uint value_dimension(uint i) const
    { return element->value_dimension(i); }

    uint num_sub_elements() const
    { return element->num_sub_elements(); }

    uint space_dimension() const
    { return element->space_dimension(); }

    void interpolate_vertex_values(double* vertex_values, double* coefficients, const ufc::cell& cell) const
    { element->interpolate_vertex_values(vertex_values, coefficients, cell); }

    void evaluate_basis(uint i, double* values, const double* x, const ufc::cell& cell) const
    { element->evaluate_basis(i, values, x, cell); }
  
    double evaluate_dof(uint i, const ufc::function& function, const ufc::cell& cell) const
    { return element->evaluate_dof(i, function, cell); }

    FiniteElement* create_sub_element(uint i) const
    { return new FiniteElement(element->create_sub_element(i)); }

    ufc::finite_element& ufc_element() const
    { return *element; } 

  private:

    // UFC finite element
    std::tr1::shared_ptr<ufc::finite_element> element;
  };

}
#endif
