// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2005-11-29

#ifndef __FORM_H
#define __FORM_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Function.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/BLASFormData.h>

namespace dolfin
{
  
  class Form
  {
  public:

    /// Constructor
    Form(uint num_functions);

    /// Destructor
    virtual ~Form();

    /// Update map to current cell
    void update(AffineMap& map);

    /// Friends
    friend class FEM;

  protected:

    // Add function
    void add(Function& f, FiniteElement* element);

    // Update coefficients
    void updateCoefficients(AffineMap& map);

    // List of finite elements for functions (coefficients)
    Array<FiniteElement*> elements;

    // List of functions (coefficients)
    Array<Function*> functions;
    
    // Coefficients of functions projected to current element
    real** c;

    // Number of functions
    uint num_functions;

    // BLAS form data
    BLASFormData blas;

  };

}

#endif
