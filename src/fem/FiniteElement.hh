// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FINITE_ELEMENT_HH
#define __FINITE_ELEMENT_HH

#include <kw_constants.h>
#include <dolfin/Grid.hh>

class ShapeFunction;
class FunctionSpace;

enum ElementType { trilin, tetlin };

class FiniteElement{
public:
  
  FiniteElement(Dolfin::Grid *grid, int no_eq, ElementType element_type);
  ~FiniteElement();

  void Update(int cellnumber);

  int GetSpaceDim();
  int GetDim();
  int GetCellNumber();
  int GetNoCells();
  int GetGlobalDof(int dof);
  int GetNoNodes();
  real GetVolume();
  real GetCircumRadius();
  
  real Coord(int node, int dim);

  friend class FunctionSpace;
  friend class ShapeFunction;
  friend class LocalField;

  FunctionSpace* GetFunctionSpace();
  ShapeFunction* GetShapeFunction(int dof, int component);

  friend class TriLinSpace;
  friend class TetLinSpace;

protected:

  real **coord;
  Grid *grid;

private:
  
  void ComputeGeometry();
  void ReallocCoord(int new_no_nodes);
  void GetCoordinates();
  
  FunctionSpace *functionspace;
  
  int nsd;
  int no_nodes;
  int cellnumber;

  real volume,circum_radius;
};

#endif
