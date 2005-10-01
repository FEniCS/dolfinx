// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2005-05-29

#ifndef __FORM_H
#define __FORM_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Function.h>
#include <dolfin/AffineMap.h>
#include <dolfin/FiniteElement.h>

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
    void update(const AffineMap& map);

    /// Friends
    friend class FEM;

  protected:

    // Add function
    void add(Function& function, const FiniteElement* element);

    // Update coefficients
    void updateCoefficients(const AffineMap& map);

    // Initialize form data for BLAS
    void initBLAS(const char* filename);

    // List of finite elements for functions (coefficients)
    Array<const FiniteElement*> elements;

    // List of functions (coefficients)
    Array<Function*> functions;
    
    // Coefficients of functions projected to current element
    real** c;

    // Number of functions
    uint num_functions;

    // Form data for BLAS
    real* blas_A;
    real* blas_G;

  };

}

#endif
