// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TET_LIN_FUNCTION_HH
#define __TET_LIN_FUNCTION_HH

#include <kw_constants.h>
#include "ShapeFunction.hh"

class LocalField;
class TetLinSpace;

class TetLinFunction: public ShapeFunction{
public:

  TetLinFunction(FunctionSpace *functionspace, int dof);
  ~TetLinFunction();
  
  void barycentric(real *point, real *bcoord);

  friend class TetLinSpace;

  // Operators
  
  real operator* (const ShapeFunction &v) const;
  real operator* (real a) const;
  
private:

};

#endif
