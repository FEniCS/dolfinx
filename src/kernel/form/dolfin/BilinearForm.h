// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/Form.h>

namespace dolfin
{

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
    
  };

}

#endif
