// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SHAPEFUNCTION_HH
#define __SHAPEFUNCTION_HH

#include <kw_constants.h>

class FunctionSpace;

class ShapeFunction{
  
public:

  ShapeFunction(FunctionSpace *functionspace, int dof);
  ~ShapeFunction();

  void SetDof(int dof);
  int  GetDof() const; 
  int  GetCellNumber();

  int  GetDim();
  int  GetNoCells();
  
  real GetCircumRadius();

  real GetCoord(int node, int dim);

  virtual void barycentric(real *point, real *bcoord) = 0;
  
  real dx;
  real dy;
  real dz;
  
  friend class FunctionSpace;
  friend class FiniteElement;
  
  void Active(bool active);
  bool Active() const;

  void Display();
  
  FiniteElement* GetFiniteElement();

  // Operators
  
  virtual real operator* (const ShapeFunction &v) const = 0;
  virtual real operator* (real a) const = 0;
  
  friend real operator* (real a, ShapeFunction &v);
  
  friend class TetLinSpace;
  friend class TetLinFunction;
  friend class TriLinFunction;
  
protected:
  
  FunctionSpace *functionspace;

  int dof;
  bool active;
  
private: 

};

#endif
