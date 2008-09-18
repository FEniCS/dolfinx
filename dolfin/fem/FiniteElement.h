// Copyright (C) 2008 Anders Logg and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-16

#ifndef __FINITE_ELEMENT_H
#define __FINITE_ELEMENT_H

#include <tr1/memory>
#include <ufc.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/elements/ElementLibrary.h>
#include <dolfin/fem/UFCCell.h>

namespace dolfin
{

  /// This is a wrapper for a UFC finite element (ufc::finite_element).

  class FiniteElement
  {
  public:

    /// Create FiniteElement. Object owns ufc::finite_element
    FiniteElement(ufc::finite_element* element) : element(element) 
    { /* Do nothing */ }

    /// Create FiniteElement. ufc_element may be shared
    FiniteElement(std::tr1::shared_ptr<ufc::finite_element> element) : element(element)
    { /* Do nothing */ }

    /// Create FiniteElement from a signature
    FiniteElement(std::string signature) : element(ElementLibrary::create_finite_element(signature))
    { /* Do nothing */ }

    std::string signature() const
    { return element->signature(); }

    uint valueRank() const
    { return element->value_rank(); }
   
    uint valueDimension(uint i) const
    { return element->value_dimension(i); }

    uint numSubElements() const
    { return element->num_sub_elements(); }

    uint spaceDimension() const
    { return element->space_dimension(); }

    void interpolateVertexValues(real* vertex_values, real* coefficients, const UFCCell& cell) const
    { element->interpolate_vertex_values(vertex_values, coefficients, cell); }

    void evaluateBasis(uint i, real* values, const real* x, const UFCCell& cell) const
    { element->evaluate_basis(i, values, x, cell); }
  
    real evaluateDof(uint i, const ufc::function& function, const ufc::cell& cell) const
    { return element->evaluate_dof(i, function, cell); }

    FiniteElement* createSubElement(uint i) const
    { return new FiniteElement(element->create_sub_element(i)); }

    ufc::finite_element& ufc_element() const
    { return *element; } 

  private:

    std::tr1::shared_ptr<ufc::finite_element> element;
  };

}
#endif
