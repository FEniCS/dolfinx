// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Galerkin.hh"
#include "FiniteElement.hh"
#include "ShapeFunction.hh"
#include "Equation.hh"
#include "EquationSystem.hh"
#include <Settings.hh>

//-----------------------------------------------------------------------------
Galerkin::Galerkin()
{
  size = grid->noNodes();
  noeq = equation->size();
  dim  = equation->dim();
  
  bc_function = settings->bc_function;
  if ( !bc_function )
	 display->Error("Boundary conditions are not specified.");
  
  // Get space dimension
  settings->Get("space dimension",&nosd);
}
//-----------------------------------------------------------------------------
Galerkin::~Galerkin()
{
  
}
//-----------------------------------------------------------------------------
void Galerkin::assemble(Grid &grid, Equation &equation, Matrix &A, Vector &b)
{
  assembleLHS(grid, equation, A);
  assembleRHS(grid, equation, b);
  setBC(grid, A,b);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleLHS(Grid &grid, Equation &equation, Matrix &A)
{
  int i,j;
  real volume;
  real integral;

  // Reallocate the matrix if necessary
  alloc(A);
  
  // Reset the matrix
  A = 0.0;
  
  // Choose the type of element, so far we only have two choices
  FiniteElement *trial;
  FiniteElement *test;
  FiniteElement *coeff;

  if ( nosd == 3 ){
    trial = new FiniteElement(grid,no_eq,tetlin);
    test  = new FiniteElement(grid,no_eq,tetlin);
    coeff = new FiniteElement(grid,no_eq,tetlin);
  }
  else{
    tria  = new FiniteElement(grid,no_eq,trilin);
    test  = new FiniteElement(grid,no_eq,trilin);
    coeff = new FiniteElement(grid,no_eq,trilin);
  }
  
  ShapeFunction *u;
  ShapeFunction *v;
  
  int no_dof;
  
  // Go through all elements and assemble
  for (CellIterator c(grid); !c.end(); ++c) {
	 
    // Update elements
    trial->update(c);
    test->update(c);
    field->update(c);
	 
    // Update equation
    equation->updateLHS(element_field);
	 
    // Get the number of dof
    no_dof = element_trial->GetDim();
    
    volume = element_trial->GetVolume();
	 
    // Compute element matrix and put the entries directly into A
    for (int test = 0; test < no_dof; test++) {
      for (int trial = 0; trial < no_dof; trial++) {
		  for (int k = 0; k < noeq; k++) {
			 for (int l = 0; l < noeq; l++) {
				
				u = element_trial->GetShapeFunction(trial,l);
				v = element_test->GetShapeFunction(test,k);
				
				if ( no_eq == 1 )
				  integral = equation->IntegrateLHS(u[0],v[0]);
				else
				  integral = ((EquationSystem *) equation)->IntegrateLHS(u,v);
				
				i = element_test->GetGlobalDof(test)*no_eq + k;
				j = element_trial->GetGlobalDof(trial)*no_eq + l;
				
				A->Add(i,j,integral*volume);
						
			 }
		  }
      }
    }
  }

  // Clean up
  delete trial;
  delete test;
  delete coeff;
  
  // Drop zero elements from matrix
  //A->DropZero(1e-12);
}
//-----------------------------------------------------------------------------
void Galerkin::assembleRHS(Grid &grid, Equation &equation, Vector &b)
{
  Cell *c;
  int i,j;

  // Allocate the vector if we have to
  alloc(b);
  
  // Reset the matrix
  b->SetToConstant(0.0);
  
  real volume;
  real integral;

  // Choose type of finite elements
  FiniteElement *element_test;
  FiniteElement *element_field;
  if ( space_dimension == 3 ){
	 element_test  = new FiniteElement(grid,no_eq,tetlin);
	 element_field = new FiniteElement(grid,no_eq,tetlin);
  }
  else{
	 element_test  = new FiniteElement(grid,no_eq,trilin);
	 element_field = new FiniteElement(grid,no_eq,trilin);
  }

  ShapeFunction *v;

  int no_dof;

  // Go through all elements and assemble
  for (int cell=0;cell<grid->GetNoCells();cell++){

    // Update elements
    element_test->Update(cell);
	 element_field->Update(cell);
    
    // Update equation
    equation->UpdateRHS(element_field);
    
    // Get the number of dof
    no_dof = element_test->GetDim();
    
    volume = element_test->GetVolume();
    
    for (int test=0;test<no_dof;test++){
      for (int k=0;k<no_eq;k++){
	
		  v = element_test->GetShapeFunction(test,k);
		  
		  if ( no_eq == 1 )
			 integral = equation->IntegrateRHS(v[0]);
		  else
			 integral = ((EquationSystem *) equation)->IntegrateRHS(v);

		  i = element_test->GetGlobalDof(test)*no_eq + k;
		  b->Add(i,integral*volume);
	
      }
    }
  }
  
  // Clean up
  delete element_test;
  
  // Drop zero elements from matrix
  //A->DropZero(1e-12);
}
//-----------------------------------------------------------------------------
void Galerkin::setBC(Grid &grid, Matrix &A, Vector &b)
{
  if ( (A->Size(0) != no_eq*no_nodes ) || ( b->Size() != no_eq*no_nodes ) )
  	 display->Error("You must assemble the matrix before settings boundary conditions.");

  Point *p;
  dolfin_bc bc;
  
  for (int i=0;i<no_nodes;i++){

	 p  = grid->GetNode(i)->GetCoord();
	 
    for (int component=0;component<no_eq;component++){

      bc = bc_function(p->x,p->y,p->z,i,component+equation->GetStartVectorComponent());
      
      switch ( bc.type ){
      case dirichlet:
		  A->SetRowIdentity(i*no_eq+component);
		  b->Set(i*no_eq+component,bc.val);
		  break;
      case neumann:
		  if ( bc.val != 0.0 )
			 display->Error("Inhomogeneous Neumann boundary conditions not implemented.");
		  break;
      default:
		  display->InternalError("Galerkin::SetBoundaryConditions()",
										 "Unknown boundary condition type.");
      }
    }

  }

}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Matrix &A)
{
  int ncols[size];
  int ii;
  bool ok = true;
  
  // Compute the number of columns for every row in the matrix
  for(int i=0;i<no_nodes;i++)
	 for (int j=0;j<no_eq;j++){
		ii = i*no_eq+j;
		ncols[ii] = no_eq * grid->GetNode(i)->GetNoNodeNeighbors();

		if ( ncols[ii] != A->GetRowLength(ii) )
		  ok = false;
		
	 }
  
  // Reallocate the matrix with the new size
  if ( !ok )
	 A->resize(size,size,ncols);
}
//-----------------------------------------------------------------------------
void Galerkin::alloc(Vector &b)
{
  if ( b.size() != size )
	 b.resize(size);
}
//-----------------------------------------------------------------------------
