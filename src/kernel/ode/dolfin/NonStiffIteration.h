// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NON_STIFF_ITERATION_H
#define __NON_STIFF_ITERATION_H

#include <dolfin/NewArray.h>
#include <dolfin/FixedPointIteration.h>

namespace dolfin
{
  class TimeSlab;
  class Element;

  /// Non-stiff fixed point iteration.

  class NonStiffIteration
  {
  public:
    
    static void stabilize(TimeSlab& timeslab, 
			  const FixedPointIteration::Residuals& r);

    static void stabilize(NewArray<Element*>& elements,
			  const FixedPointIteration::Residuals& r);
    
    static void stabilize(Element& element,
			  const FixedPointIteration::Residuals& r);

    static void update(Element& element);

  };

}

#endif
