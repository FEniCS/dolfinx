// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LINEAR_FORM_H
#define __LINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

  class NewFiniteElement;
  
  class LinearForm : public Form
  {
  public:
    
    /// Constructor
    LinearForm(const NewFiniteElement& element);
    
    /// Destructor
    virtual ~LinearForm();

    /// Compute element vector (interior contribution)
    virtual bool interior(real* b) const;

    /// Compute element vector (boundary contribution)
    virtual bool boundary(real* b) const;
    
  };

}

#endif
