// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LINEAR_FORM_H
#define __LINEAR_FORM_H

#include <dolfin/constants.h>
#include <dolfin/Form.h>

namespace dolfin
{

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms
  
  class LinearForm : public Form
  {
  public:
    
    /// Constructor
    LinearForm();
    
    /// Destructor
    ~LinearForm();

    /// Evaluation of bilinear form
    real operator() (TestFunction v);
    
  };

}

#endif
