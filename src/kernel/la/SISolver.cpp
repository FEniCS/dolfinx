// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include "SISolver.hh"
#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SISolver::SISolver()
{
  iterative_method = gaussseidel;
  
  max_no_iterations = 100;

  tol = 1.0e-6;
}
//-----------------------------------------------------------------------------
void SISolver::Solve(SparseMatrix* A, Vector* x, Vector* b)
{
  if (A->Size(0)!=A->Size(1)) display->Error("Must be a square matrix.");

  if (A->Size(0)!=b->Size())  display->Error("Not compatible matrix and vector sizes.");
  
  if (x->Size()!=b->Size())
    x->Resize(b->Size());

  real norm_b = b->Norm();
  if ( norm_b < DOLFIN_EPS ){
    x->SetToConstant(0.0);
    return;
  }  
  
  residual = 2.0*tol*norm_b;

  iteration = 0;
   while ( residual/norm_b > tol ){
    iteration ++;
    switch( iterative_method ){ 
    case richardson:
      IterateRichardson(A,x,b);
      break;
    case jacobi:
      IterateJacobi(A,x,b);
      break;
    case gaussseidel:
      IterateGaussSeidel(A,x,b);
      break;
    case sor:
      IterateSOR(A,x,b);
      break;
    default:
      display->InternalError("SISolver::Solve()","Iterative method not implemented");
    }
    ComputeResidual(A,x,b);
    display->Status(2,"Iteration no %i: res/||b|| = %f",iteration,residual/norm_b);
  }
  
}
//-----------------------------------------------------------------------------
void SISolver::SetNoIterations( int noit )
{
  max_no_iterations = noit;
}
//-----------------------------------------------------------------------------
void SISolver::SetMethod( Method mtd )
{
  iterative_method = mtd;
}
//-----------------------------------------------------------------------------
void SISolver::IterateRichardson(SparseMatrix* A, Vector* x, Vector* b)
{
  real aii,aij,norm_b,Ax;
  
  int j;

  Vector x0( x->Size() );
  x0.CopyFrom(x);

  for (int i=0; i<A->Size(0); i++){
    x->Set(i,0.0);
    for (int pos=0; pos<A->GetRowLength(i); pos++){
      aij = A->Get(i,&j,pos);
		if ( j == -1 )
		  break;
      if (i==j){
		  x->Add(i,(1.0-aij)*x0(j));
      } else{
		  x->Add(i,-aij*x0(j));
      }	  
    }
    x->Add(i,(*b)(i));
  }

}
//-----------------------------------------------------------------------------
void SISolver::IterateJacobi(SparseMatrix* A, Vector* x, Vector* b)
{
  real aii,aij,norm_b,Ax;
  int j;

  Vector x0( x->Size() );
  x0.CopyFrom(x);

  for (int i=0; i<A->Size(0); i++){
    x->Set(i,0.0);
    for (int pos=0; pos<A->GetRowLength(i); pos++){
      aij = A->Get(i,&j,pos);
      if ( j == -1 ) break;
      if (i==j) aii = aij;
      else x->Add(i,-aij*x0(j));
    }
    x->Add(i,(*b)(i));
    x->Mult(i,1.0/aii);
  }
}
//-----------------------------------------------------------------------------
void SISolver::IterateGaussSeidel(SparseMatrix* A, Vector* x, Vector* b)
{
  real aii,aij,Ax;
  
  int j;

  for (int i=0; i<A->Size(0); i++){
    x->Set(i,0.0);
    for (int pos=0; pos<A->GetRowLength(i); pos++){
      aij = A->Get(i,&j,pos);
		if ( j == -1 )
		  break;
      if (j==i){
		  aii = aij;
      } else{
		  x->Add(i,-aij*(*x)(j));
      }	  
    }
    x->Add(i,(*b)(i));
    x->Mult(i,1.0/aii);
  }

}
//-----------------------------------------------------------------------------
void SISolver::IterateSOR(SparseMatrix* A, Vector* x, Vector* b)
{
  real aii,aij,norm_b,Ax;
  
  int j;

  real omega = 1.0;

  for (int i=0; i<A->Size(0); i++){
    x->Set(i,0.0);
    for (int pos=0; pos<A->GetRowLength(i); pos++){
      aij = A->Get(i,&j,pos);
		if ( j == -1 )
		  break;
      if (j==i){
	aii = aij;
	x->Add(i,(1.0-omega)*aii*(*x)(j));
      } else{
	x->Add(i,-omega*aij*(*x)(j));
      }	  
    }
    x->Add(i,(*b)(i));
    x->Mult(i,1.0/aii);
  }

}
//-----------------------------------------------------------------------------
void SISolver::ComputeResidual(SparseMatrix* A, Vector* x, Vector* b)
{
  residual = 0.0;
  real Ax;
  
  for (int i=0;i<A->Size(0);i++){
	 Ax = A->Mult(i,x);
	 residual += sqr((*b)(i)-Ax);
  }

  residual = sqrt(residual);
}
//-----------------------------------------------------------------------------
