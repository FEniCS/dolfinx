// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __P1TRI_ELEMENT_H
#define __P1TRI_ELEMENT_H

#include <dolfin/FiniteElement.h>
#include <dolfin/P1Tri.h>

namespace dolfin {
  
  class P1TriElement : public FiniteElement {
  public:

	 P1TriElement() : FiniteElement(trial, test) {}
	 
  private:

	 P1Tri trial;
	 P1Tri test;
	 
  };
  
}

#endif
