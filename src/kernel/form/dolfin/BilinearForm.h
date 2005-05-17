// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

  class AffineMap;
  class FiniteElement;

  /// BilinearForm represents a bilinear form a(v, u) with arguments v
  /// and u basis functions of the finite element space defined by a
  /// pair of finite elements (test and trial).

  class BilinearForm : public Form
  {
  public:
    
    /// Constructor
    BilinearForm(uint num_functions = 0);
    
    /// Destructor
    virtual ~BilinearForm();
    
    /// Compute element matrix (interior contribution)
    virtual void eval(real block[], const AffineMap& map) const;
    
    /// Compute element matrix (boundary contribution)
    virtual void eval(real block[], const AffineMap& map, uint boundary) const;

    /// Return finite element defining the test space
    const FiniteElement& test() const;

    /// Return finite element defining the trial space
    const FiniteElement& trial() const;

  protected:

    // Finite element defining the test space
    FiniteElement* _test;

    // Finite element defining the trial space
    FiniteElement* _trial;
    
  };

}

#endif
