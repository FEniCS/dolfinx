// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Grid.hh"
#include "Cell.hh"
#include <Display.hh>
#include <Settings.hh>
#include "FiniteElement.hh"
#include "FunctionSpace.hh"
#include "ShapeFunction.hh"
#include "TriLinSpace.hh"
#include "TetLinSpace.hh"

extern Settings *settings;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(Grid *grid, int no_eq, ElementType element_type)
{
  settings->Get("space dimension", &nsd);
  this->grid = grid;
  
  // Allocate for nodes
  coord = 0;
  ReallocCoord(grid->GetCell(0)->GetSize());

  // Choose function space
  switch ( element_type ){
  case trilin:
	 functionspace = new TriLinSpace(this,no_eq);
	 break;
  case tetlin:
	 functionspace = new TetLinSpace(this,no_eq);
	 break;
  default:
	 display->InternalError("FiniteElement::FiniteElement()","Unknown element type: %d",element_type);
  }

  cellnumber = -1;
}
//-----------------------------------------------------------------------------
FiniteElement::~FiniteElement()
{
  for (int i=0;i<no_nodes;i++)
    delete [] coord[i];
  delete [] coord;
  coord = 0;
  
  delete functionspace;
}
//-----------------------------------------------------------------------------
void FiniteElement::Update(int cellnumber)
{
  if (this->cellnumber != cellnumber){

    this->cellnumber = cellnumber;
    
    // Check if the number of nodes has changed
    int new_no_nodes;
    if ( (new_no_nodes=grid->GetCell(cellnumber)->GetSize()) != no_nodes )
      ReallocCoord(new_no_nodes);
    
    // Get the coordinates
    GetCoordinates();

	 // Compute geometric data
    ComputeGeometry();

	 // Update the functionspace
    functionspace->Update();
  }
}
//-----------------------------------------------------------------------------
FunctionSpace*  FiniteElement::GetFunctionSpace()
{
  return functionspace;
}
//-----------------------------------------------------------------------------
ShapeFunction* FiniteElement::GetShapeFunction(int dof, int component) 
{
  ShapeFunction *shapefunction = functionspace->shapefunction[dof];

  for (int i=0;i<functionspace->nvc;i++)
    if ( i == component ){
      shapefunction[i].active = true;

		// FIXME: Maybe this should not be done here
		shapefunction[i].dx = functionspace->gradient[dof][0];
		shapefunction[i].dy = functionspace->gradient[dof][1];
		shapefunction[i].dz = functionspace->gradient[dof][2];
	 }
	 else{
      shapefunction[i].active = false;

		// FIXME: Maybe this should not be done here
		shapefunction[i].dx = 0.0;
		shapefunction[i].dy = 0.0;
		shapefunction[i].dz = 0.0;
	 }

  return shapefunction;
}
//-----------------------------------------------------------------------------
int FiniteElement::GetSpaceDim()
{
  return nsd;
}
//-----------------------------------------------------------------------------
int FiniteElement::GetGlobalDof(int dof)
{
  // FIXME: Only for nodal basis
  return grid->GetCell(cellnumber)->GetNode(dof)->GetNodeNo();
}
//-----------------------------------------------------------------------------
int FiniteElement::GetDim()
{
  return functionspace->dim;
}
//-----------------------------------------------------------------------------
int FiniteElement::GetCellNumber()
{
  return cellnumber;
}
//-----------------------------------------------------------------------------
int FiniteElement::GetNoCells()
{
  return grid->GetNoCells();
}
//-----------------------------------------------------------------------------
int FiniteElement::GetNoNodes()
{
  return no_nodes;
}
//-----------------------------------------------------------------------------
real FiniteElement::GetVolume()
{
  return volume;
}
//-----------------------------------------------------------------------------
real FiniteElement::GetCircumRadius()
{
  return circum_radius;
}
//-----------------------------------------------------------------------------
void FiniteElement::ComputeGeometry()
{
  Cell *c;
  c = grid->GetCell(cellnumber);
  
  volume = c->ComputeVolume(grid);

  circum_radius = c->ComputeCircumRadius(grid,volume);
}
//-----------------------------------------------------------------------------
void FiniteElement::ReallocCoord(int new_no_nodes)
{
  if ( coord ){
	 for (int i=0;i<no_nodes;i++)
		delete [] coord[i];
	 delete [] coord;
	 coord = 0;
  }

  coord = new (real *)[new_no_nodes];
  for (int i=0;i<new_no_nodes;i++)
	 coord[i] = new real[nsd];

  no_nodes = new_no_nodes;
}
//-----------------------------------------------------------------------------
void FiniteElement::GetCoordinates()
{
  Cell *c = grid->GetCell(cellnumber);
  Node *n;
  
  for (int i=0;i<no_nodes;i++){
	 n = c->GetNode(i);
	 for (int j=0;j<nsd;j++)
		coord[i][j] = n->GetCoord(j);
  }
  
}
//-----------------------------------------------------------------------------
