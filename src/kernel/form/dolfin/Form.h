// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __FORM_H
#define __FORM_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/FiniteElement.h>

namespace dolfin
{

  class Cell;
  class Function;
  
  class Form
  {
  public:

    /// Constructor
    Form(uint num_functions);

    /// Destructor
    virtual ~Form();

    /// Update map to current cell
    void update(const Cell& cell);

    /// Friends
    friend class FEM;

  protected:

    // Update coefficients
    void updateCoefficients(const Cell& cell);

    // Add function
    void add(Function& function, const FiniteElement* element);

    // List of finite elements for functions (coefficients)
    Array<const FiniteElement*> elements;

    // List of functions (coefficients)
    Array<const Function*> functions;
    
    // Coefficients of functions projected to current element
    real** w;

    // Number of functions
    uint num_functions;

  };

}

#endif
