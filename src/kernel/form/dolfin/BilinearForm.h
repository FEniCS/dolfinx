// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

  class NewFiniteElement;

  /// BilinearForm represents a bilinear form a(v, u) with arguments v
  /// and u basis functions of the finite element space defined by a
  /// pair of finite elements (test and trial).

  class BilinearForm : public Form
  {
  public:
    
    /// Constructor
    BilinearForm();
    
    /// Destructor
    virtual ~BilinearForm();
    
    /// Compute element matrix (interior contribution)
    virtual bool interior(real* block) const;
    
    /// Compute element matrix (boundary contribution)
    virtual bool boundary(real* block) const;

    /// Return finite element defining the test space
    const NewFiniteElement& test() const;

    /// Return finite element defining the trial space
    const NewFiniteElement& trial() const;

  protected:

    // Finite element defining the test space
    NewFiniteElement* _test;

    // Finite element defining the trial space
    NewFiniteElement* _trial;
    
  };

}

#endif
