// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TETLIN_SPACE_HH
#define __TETLIN_SPACE_HH

#include "FunctionSpace.hh"

class TetLinFunction;
class LocalField;
class TriLinFunction;

namespace Dolfin{ class TetLinSpace: public FunctionSpace{
public:
  
  TetLinSpace(FiniteElement *element, int nvc);
  ~TetLinSpace();

  void Update();
  
  friend class TetLinFunction;
  friend class FiniteElement;

private:

  real IntShapeFunction(int m1);
  real IntShapeFunction(int m1, int m2);
  real IntShapeFunction(int m1, int m2, int m3);
  real IntShapeFunction(int m1, int m2, int m3, int m4);

  real j11,j12,j13,j21,j22,j23,j31,j32,j33;
  real det,d;
  
}; }
  
#endif
