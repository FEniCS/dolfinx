// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "LocalField.hh"
#include "GlobalField.hh"
#include "FunctionSpace.hh"
#include "FiniteElement.hh"
#include "ShapeFunction.hh"
#include <Settings.hh>

extern Settings *settings;

using namespace Dolfin;

//-----------------------------------------------------------------------------
LocalField::LocalField()
{
  // Local field is not initialised until explicitly updated
  dim               = 0;
  dof               = 0;
  functionspace     = 0;
  shapefunction     = 0;
  component         = 0;
  globalfield       = 0;

  dx = 0.0;
  dy = 0.0;
  dz = 0.0;
}
//-----------------------------------------------------------------------------
LocalField::~LocalField()
{
  if ( dof )
	 delete [] dof;
  dof = 0;
}
//----------------------------------------------------------------------------- 
void LocalField::AttachGlobalField(GlobalField *globalfield, int component)
{
  this->globalfield = globalfield;
  this->component = component;
}
//----------------------------------------------------------------------------- 
void LocalField::operator= ( LocalField &lf )
{
  if ( functionspace != lf.functionspace ){
    functionspace = lf.functionspace;
	 shapefunction = lf.shapefunction;

	 if ( dim != functionspace->dim || !dof ){
		if ( dof )
		  delete dof;
		dof = new real[dim];
		dim = functionspace->dim;
	 }

  }

  // Set the degrees of freedom
  for (int i=0;i<dim;i++)
	 dof[i] = lf.dof[i];
  
  // Compute gradient
  ComputeGradient();
} 
//-----------------------------------------------------------------------------
void LocalField::Add(LocalField &lf)
{
  if ( functionspace != lf.functionspace )
    display->InternalError("LocalField::Add()","Cannot add LocalFields from different functionspaces");

  for (int i=0;i<dim;i++)
	 dof[i] += lf.dof[i];

  dx += lf.dx;
  dy += lf.dy;
  dz += lf.dz;
}
//-----------------------------------------------------------------------------
void LocalField::Mult(real a)
{
  for (int i=0;i<dim;i++)
	 dof[i] *= a;

  dx *= a;
  dy *= a;
  dz *= a;
}
//-----------------------------------------------------------------------------
void LocalField::SetToConstant(real a)
{
  for (int i=0;i<dim;i++)
	 dof[i] = a;

  dx = 0.0;
  dy = 0.0;
  dz = 0.0;
}
//-----------------------------------------------------------------------------
void LocalField::Mean(LocalField &v, LocalField &w)
{
  if ( v.functionspace != w.functionspace )
	 display->InternalError("LocalField::Mean()",
									"Function spaces don't match.");
  
  if ( functionspace != v.functionspace ){

    functionspace = v.functionspace;
	 shapefunction = v.shapefunction;
    int new_dim = functionspace->dim;

	 if ( new_dim != dim ){
		dim = new_dim;
		if ( dof )
		  delete dof;
		dof = new real[dim];
	 }

  }
	 
  for (int i=0;i<dim;i++)
	 dof[i] = 0.5*(v.dof[i] + w.dof[i]);

  dx = 0.5*(v.dx+w.dx);
  dy = 0.5*(v.dy+w.dy);
  dz = 0.5*(v.dz+w.dz);
}
//-----------------------------------------------------------------------------
const ShapeFunction* LocalField::GetShapeFunction(int i) const 
{
  return shapefunction[i];
}
//-----------------------------------------------------------------------------
real LocalField::GetCoord(int node, int dim) const
{
  return functionspace->GetCoord(node,dim);
}
//-----------------------------------------------------------------------------
real LocalField::GetDofValue(int i) const
{
  return dof[i];
}
//-----------------------------------------------------------------------------
real LocalField::GetMeanValue() const
{
  real mean = 0.0;
  for (int i=0;i<dim;i++) mean += dof[i];

  return mean/real(dim);
}
//-----------------------------------------------------------------------------
int LocalField::GetDim() const
{
  return dim;
}
//-----------------------------------------------------------------------------
int LocalField::GetCellNumber() const
{
  return functionspace->GetCellNumber();
}
//-----------------------------------------------------------------------------
void LocalField::Update(FiniteElement *element, real t)
{
  if ( !globalfield )
	 display->Error("Unable to update localfield, no global field attached.");

  // Reallocate degrees of freedom for the new function space if necessary
  Resize(element->GetFunctionSpace());

  // Update the degrees of freedom
  globalfield->GetLocalDof(element->GetCellNumber(),t,dof,component);

  // Compute gradient
  ComputeGradient();
}
//-----------------------------------------------------------------------------
void LocalField::Display()
{
  printf("LocalField v = [");
  for (int i=0;i<(dim-1);i++)
	 printf("%f ",dof[i]);
  printf("%f]\n",dof[dim-1]);

  //printf("Active: ");
  // for (int i=0;i<dim;i++)
  // if ( GetShapeFunction(i)->Active() )
  //		printf("1 ");
  //	 else
  //	printf("0 ");
  //printf("\n");
}
//-----------------------------------------------------------------------------
FunctionSpace* LocalField::GetFunctionSpace()
{
  return functionspace;
}
//-----------------------------------------------------------------------------
FiniteElement* LocalField::GetFiniteElement() const
{
  return functionspace->GetFiniteElement();
}
//-----------------------------------------------------------------------------
real LocalField::operator* (const LocalField &v) const
{
  real sum_i = 0.0;
  real sum = 0.0;
  
  for (int i=0;i<dim;i++){
    sum_i = 0.0;
    for (int j=0;j<v.dim;j++)
      sum_i += v.dof[j] * ((*shapefunction[i])*(*v.shapefunction[j]));
    sum += dof[i] * sum_i;
  }

  return sum;
}
//-----------------------------------------------------------------------------
real LocalField::operator* (const ShapeFunction &v) const
{
  if ( !v.Active() ) return 0.0;

  real sum = 0.0;

  for (int i=0;i<dim;i++)
	 sum += ( dof[i] * ( (*shapefunction[i])*v ) );
  
  return sum;
}
//-----------------------------------------------------------------------------
real LocalField::operator* (real a) const
{
  real sum = 0.0;

  for (int i=0;i<dim;i++)
	 sum += (*shapefunction[i]) * dof[i];
  
  return a*sum;
}
//-----------------------------------------------------------------------------
real operator* (real a, LocalField &v)
{
  // This makes sure that a * LocalField commutes

  return ( v*a );
}
//-----------------------------------------------------------------------------
real operator* (const ShapeFunction &v, const LocalField &w)
{
  // This makes sure that ShapeFunction * LocalField commutes

  return ( w*v );
}
//-----------------------------------------------------------------------------
void LocalField::Resize(FunctionSpace *functionspace)
{
  // Resize if necessary

  int new_dim = functionspace->dim;
  
  if ( new_dim != dim ){
	 if ( dof )
		delete [] dof;
	 dof = new real[new_dim];
	 this->dim  = new_dim;
  }

  // FIXME: This shouldn't be here?
  this->functionspace = functionspace;
  this->shapefunction = functionspace->shapefunction;
  this->nsd           = functionspace->nsd;
}
//-----------------------------------------------------------------------------
void LocalField::ComputeGradient()
{
  // Update gradient
  dx = 0.0;
  dy = 0.0;
  dz = 0.0;
  
  for (int j=0;j<dim;j++) dx += dof[j]*functionspace->gradient[j][0];
  for (int j=0;j<dim;j++) dy += dof[j]*functionspace->gradient[j][1];
  if ( nsd > 2 )
	 for (int j=0;j<dim;j++) dz += dof[j]*functionspace->gradient[j][2];
}
//-----------------------------------------------------------------------------
