// Copyright (C) 2008 Johan Jansson
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-03-18
// Last changed: 2008-03-18

#ifndef __PROJECTL2_H
#define __PROJECTL2_H

#include <dolfin/mesh/Mesh.h>
#include "Function.h"
#include <ufc.h>

namespace dolfin
{

  /// Compute L2 projection fB of fA on FEM space element
  void projectL2(Mesh& meshB, Function& fA, Function& fB,
		 ufc::finite_element& element);
  
  /// Compute L2 projection fB of fA (discrete function) on FEM space element
  void projectL2NonMatching(Mesh& meshB, Function& fA, Function& fB,
			    ufc::finite_element& element);
  
  /// Represent discrete function fA as pointwise user-defined function
  class NonMatchingFunction : public Function
  {
  public:
    
    NonMatchingFunction(Mesh& mesh, Function& fA);
    
    virtual void eval(real* values, const real* x) const;
    
    Function& fA;
  };
  
}

#endif
