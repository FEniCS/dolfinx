// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRI_LIN_FUNCTION_HH
#define __TRI_LIN_FUNCTION_HH

#include <kw_constants.h>
#include "ShapeFunction.hh"

class LocalField;
class TriLinSpace;

class TriLinFunction: public ShapeFunction{
public:

  TriLinFunction(FunctionSpace *functionspace, int dof);
  ~TriLinFunction();

  void barycentric(real *point, real *bcoord);

  friend class TriLinSpace;

  // Operators
  
  real operator* (const ShapeFunction &v) const;
  real operator* (real a) const;
  
private:

  

};

#endif
