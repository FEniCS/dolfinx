// Copyright (C) 2008 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-11-20

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <dolfin/log/log.h>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <ufc.h>
#include <dolfin/common/types.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/elements/ElementLibrary.h>

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    /// Create finite element from UFC finite element (data may be shared)
    FiniteElement(boost::shared_ptr<const ufc::finite_element> element) : element(element) {}

    /// Create finite element from UFC finite element
    FiniteElement(const ufc::finite_element& element) : element(&element, NoDeleter<const ufc::finite_element>()) {}
    //FiniteElement(const ufc::finite_element& element, uint dummy) : element(&element, NoDeleter<const ufc::finite_element>()) {}

    /// Create FiniteElement from a signature
    FiniteElement(std::string signature) : element(ElementLibrary::create_finite_element(signature)) {}

    /// Destructor
    ~FiniteElement()
    {}

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
    
    /// Create sub element
    boost::shared_ptr<const FiniteElement> create_sub_element(uint i) const
    { 
      boost::shared_ptr<const ufc::finite_element>  ufc_element(element->create_sub_element(i));
      boost::shared_ptr<const FiniteElement> _element(new const FiniteElement(ufc_element));
      return _element; 
    }

    /// Return ufc::finite_element
    boost::shared_ptr<const ufc::finite_element> ufc_element() const
    { return element; } 

    /// Extract sub finite element for component
    boost::shared_ptr<const FiniteElement> extract_sub_element(const std::vector<uint>& component) const;

  private:

    // Recursively extract sub finite element
    static boost::shared_ptr<const FiniteElement> extract_sub_element(const FiniteElement& finite_element, 
                                              const std::vector<uint>& component);

    // UFC finite element
    boost::shared_ptr<const ufc::finite_element> element;

  };

}
#endif
