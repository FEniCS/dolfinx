// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ODE_FUNCTION_H
#define __ODE_FUNCTION_H

#include <dolfin/ElementData.h>
#include <dolfin/GenericFunction.h>

namespace dolfin {

  /// ODEFunction represents a solution of an ODE. Note that
  /// an ODEFunction contains its data. (Whereas a DofFunction
  /// is constructed from a given Mesh and a given Vector and
  /// only contains references to data.)

  class ODEFunction : public GenericFunction {
  public:
    
    ODEFunction(unsigned int N);
    ~ODEFunction();
	 
    // Evaluation of function
    real operator() (unsigned int i, real t);

    // Get element data
    ElementData& elmdata();

  private:
    
    ElementData _elmdata;
    
  };

}

#endif
