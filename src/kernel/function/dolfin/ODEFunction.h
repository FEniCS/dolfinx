// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_FUNCTION_H
#define __ODE_FUNCTION_H

#include <dolfin/GenericFunction.h>

namespace dolfin {

  class ElementData;
  
  class ODEFunction : public GenericFunction {
  public:
    
    ODEFunction(ElementData& elmdata);
    ~ODEFunction();
	 
    // Evaluation of function
    real operator() (unsigned int i, real t) const;

  private:
    
    ElementData& elmdata;
    
  };

}

#endif
