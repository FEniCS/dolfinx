// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BILINEAR_FORM_H
#define __BILINEAR_FORM_H

#include <dolfin/NewArray.h>
#include <dolfin/IndexPair.h>
#include <dolfin/Form.h>

namespace dolfin
{

  class NewFiniteElement;
  
  class BilinearForm : public Form
  {
  public:
    
    /// Constructor
    BilinearForm(const NewFiniteElement& element);
    
    /// Destructor
    virtual ~BilinearForm();
    
    /// Compute element matrix (interior contribution)
    virtual bool interior(real** A) const;
    
    /// Compute element matrix (boundary contribution)
    virtual bool boundary(real** A) const;
    
    /// Friends
    friend class NewFEM;

  protected:
    
    //List of nonzero indices
    NewArray<IndexPair> nonzero;

  };

}

#endif
