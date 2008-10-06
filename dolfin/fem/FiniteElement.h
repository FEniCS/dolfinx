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
#include <dolfin/elements/ElementLibrary.h>

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    // FIXME: This constructor should be removed!
    /// Create finite element from UFC finite element pointer
    FiniteElement(ufc::finite_element* element) : element(element) {}

    // FIXME: This constructor should be removed!
    /// Create finite element from UFC finite element
    FiniteElement(ufc::finite_element& element, uint dummy) : element(&element, NoDeleter<ufc::finite_element>()) {}

    // FIXME: This constructor should be added!
    /// Create finite element from UFC finite element
    //FiniteElement(const ufc::finite_element& element, uint dummy) : element(&element, NoDeleter<const ufc::finite_element>()) {}

    /// Create finite element from UFC finite element (data may be shared)
    FiniteElement(std::tr1::shared_ptr<ufc::finite_element> element) : element(element) {}

    /// Create FiniteElement from a signature
    FiniteElement(std::string signature) : element(ElementLibrary::create_finite_element(signature)) {}

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
    
    // FIXME: Use shared_ptr
    // FIXNE: Return-value should be const
    FiniteElement* create_sub_element(uint i) const
    { return new FiniteElement(element->create_sub_element(i)); }

    // FIXME: Use shared_ptr
    // FIXNE: Return-value should be const
    ufc::finite_element& ufc_element() const
    { return *element; } 

  private:

    // UFC finite element
    std::tr1::shared_ptr<ufc::finite_element> element;

  };

}
#endif
