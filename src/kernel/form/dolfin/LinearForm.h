// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __LINEAR_FORM_H
#define __LINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

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
    
  };

}

#endif
