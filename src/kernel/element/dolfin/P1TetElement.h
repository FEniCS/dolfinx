// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __P1TET_ELEMENT_H
#define __P1TET_ELEMENT_H

#include <dolfin/FiniteElement.h>
#include <dolfin/P1Tet.h>

namespace dolfin {
  
  class P1TetElement : public FiniteElement {
  public:

	 P1TetElement() : FiniteElement(trial, test) {}
	 
  private:

	 P1Tet trial;
	 P1Tet test;
	 
  };
  
}

#endif
