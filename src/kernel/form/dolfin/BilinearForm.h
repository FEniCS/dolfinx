// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/constants.h>
#include <dolfin/Form.h>

namespace dolfin
{

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms

  class BilinearForm : public Form
  {
  public:
    
    /// Constructor
    BilinearForm();
    
    /// Destructor
    ~BilinearForm();

    /// Evaluation of bilinear form
    real operator() (TrialFunction u, TestFunction v);
    
  };

}

#endif
