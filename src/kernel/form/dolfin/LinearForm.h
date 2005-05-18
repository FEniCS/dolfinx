// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __LINEAR_FORM_H
#define __LINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

  /// LinearForm represents a linear form L(v) with argument v (the
  /// test function) a basis function of the finite element space
  /// defined by a finite element.

  class LinearForm : public Form
  {
  public:
    
    /// Constructor
    LinearForm(uint num_functions = 0);
    
    /// Destructor
    virtual ~LinearForm();

    /// Compute element vector (interior contribution)
    virtual void eval(real block[], const AffineMap& map) const;

    /// Compute element vector (boundary contribution)
    virtual void eval(real block[], const AffineMap& map, uint boundary) const;

    /// Return finite element defining the test space
    const FiniteElement& test() const;

  protected:

    // Finite element defining the test space
    FiniteElement* _test;

  };

}

#endif
