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
    LinearForm();
    
    /// Destructor
    virtual ~LinearForm();

    /// Compute element vector (interior contribution)
    virtual bool interior(real* block) const;

    /// Compute element vector (boundary contribution)
    virtual bool boundary(real* block) const;

    /// Return finite element defining the test space
    const NewFiniteElement& test() const;

  protected:

    // Finite element defining the test space
    NewFiniteElement* _test;

  };

}

#endif
