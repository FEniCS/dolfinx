// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "GlobalField.hh"
#include <Display.hh>
#include <Settings.hh>
#include <Grid.hh>
#include <Vector.hh>
#include <Output.hh>

//-----------------------------------------------------------------------------
GlobalField::GlobalField(Grid *grid, real constant)
{
  InitCommon(grid);

  representation = CONSTANT;
  this->constant = constant;
  
  no_dof = 1;
}
//-----------------------------------------------------------------------------
GlobalField::GlobalField(Grid *grid, Vector *x, int nvd = 1)
{
  InitCommon(grid);

  representation = NODAL;
  dof_values     = x;
  
  no_dof = x->Size()/nvd;
  this->nvd = nvd;

  if ( x->Size() % nvd != 0 )
	 display->InternalError("GlobalField:GlobalField",
									"Vector dimension (%d) does not match length of vector (%d).",nvd,x->Size());
}
//-----------------------------------------------------------------------------
GlobalField::GlobalField(Grid *grid, const char *field)
{
  InitCommon(grid);

  representation = FUNCTION;
  function       = settings->GetFunction(field);
}
//-----------------------------------------------------------------------------
GlobalField::GlobalField(GlobalField *f1, GlobalField *f2)
{
  InitCommon(0);

  if ( f1->grid != f2->grid )
	 display->InternalError("GlobalField:GlobalField()",
									"Collection of global fields must have the same grid.");

  representation = LIST;
  
  grid = f1->grid;
  list = new (GlobalField *)[2];
  listsize = 2;
 
  list[0]  = f1;
  list[1]  = f2;
}
//-----------------------------------------------------------------------------
GlobalField::GlobalField(GlobalField *f1, GlobalField *f2, GlobalField *f3)
{
  InitCommon(0);

  if ( (f1->grid != f2->grid) | (f2->grid != f3->grid) )
	 display->InternalError("GlobalField:GlobalField()",
									"Collection of global fields must have the same grid.");
  
  representation = LIST;
  
  grid = f1->grid;
  list = new (GlobalField *)[3];
  listsize = 3;
  
  list[0] = f1;
  list[1] = f2;
  list[2] = f3;
}
//-----------------------------------------------------------------------------
GlobalField::GlobalField(GlobalField **f, int n)
{
  InitCommon(0);

  if ( n <= 0 )
	 display->InternalError("GlobalField::GlobalField()",
									"The number of fields in a collection of fields must be positive.");
  for (int i=0;i<(n-1);i++)
  if ( f[i]->grid != f[i+1]->grid )
	 display->InternalError("GlobalField:GlobalField()",
									"Collection of global fields must have the same grid.");

  representation = LIST;
  
  grid = f[0]->grid;
  list = new (GlobalField *)[n];
  listsize = n;
  
  for (int i=0;i<n;i++)
	 list[i] = f[i];

  int dimensions[n];
  for (int i=0;i<n;i++)
	 dimensions[i] = f[i]->nvd;
}
//-----------------------------------------------------------------------------
GlobalField::~GlobalField()
{
  if ( output )
	 delete output;
  output = 0;

  if ( list )
	 delete list;
  list = 0;
}
//-----------------------------------------------------------------------------
void GlobalField::SetSize(int no_data, ...)
{
  va_list aptr;
  va_start(aptr,no_data);

  if ( output )
	 delete output;
  output = new Output(no_data,aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void GlobalField::SetLabel(const char *name, const char *label, int i = 0)
{
  if ( !output )
	 output = new Output(1,nvd);
  
  output->SetLabel(i,name,label);
}
//-----------------------------------------------------------------------------
int GlobalField::GetNoDof()
{
  return no_dof;
}
//-----------------------------------------------------------------------------
int GlobalField::GetVectorDim()
{
  return nvd;
}
//-----------------------------------------------------------------------------
int GlobalField::GetDim(int cell)
{
  // FIXME: no shapefunctions does not have to be same as no nodes 
  return ( grid->GetCell(cell)->GetSize() );
}
//-----------------------------------------------------------------------------
void GlobalField::GetLocalDof(int cell, real t, real *local_dof, int component = 0)
{
  // FIXME: no shapefunctions does not have to be same as no nodes 
  Cell *c = grid->GetCell(cell);
  int local_dim = c->GetSize();

  switch( representation ){ 
  case NODAL: 
    for (int i=0;i<local_dim;i++)
      local_dof[i] = dof_values->Get(c->GetNode(i)->GetNodeNo()*nvd + component); 
    break;
  case CONSTANT:
    for (int i=0;i<local_dim;i++) 
      local_dof[i] = constant;
    break;
  case FUNCTION:

	 // Return zero if the function is not specified
	 if ( !function ){
		for (int i=0;i<local_dim;i++)
		  local_dof[i] = 0.0;
		return;
	 }

	 // Get the value from the function
	 Point *p;
	 for (int i=0;i<local_dim;i++){
		p = c->GetNode(i)->GetCoord();
		local_dof[i] = function(p->x,p->y,p->z,t);
	 }
	 
    break;
  case LIST:
	 display->InternalError("GlobalField::GetLocalDof()","Not available for collection of fields.");
	 break;
  case NONE:
    display->InternalError("GlobalField::GetLocalDof()","GlobalField is not initialized");
	 break;
  default:
    display->InternalError("GlobalField::GetLocalDof()","Unknown representation for GlobalField: %d",representation);
  }
  
}
//-----------------------------------------------------------------------------
void GlobalField::Save()
{
  Save(0.0);
}
//-----------------------------------------------------------------------------
void GlobalField::Save(real t)
{
  // If Output is not initialised, assume the simplest possible
  if ( !output )
	 output = new Output(1,nvd);

  Vector **vectors = 0;
  
  switch ( representation ){
  case NODAL:
	 output->AddFrame(grid,&dof_values,t);
	 break;
  case LIST:
	 vectors = new (Vector *)[listsize];
	 for (int i=0;i<listsize;i++){
		if ( list[i]->representation != NODAL )
		  display->InternalError("GlobalField:Save()","Output for collection of fields only implemented for nodal representation.");
		vectors[i] = list[i]->dof_values;
	 }
	 output->AddFrame(grid,vectors,t,listsize);
	 delete vectors;
	 break;
  default:
	 int *a;
	 a[12313123] = 1;
	 
	 display->Message(0,"representation = %d",representation);
	 display->InternalError("GlobalField:Save()",
									"Output only implemented for nodal representation (possibly collections).");
  }
  
}
//-----------------------------------------------------------------------------
void GlobalField::InitCommon(Grid *grid)
{
  this->grid = grid;

  representation = NONE;
  no_dof         = 0;
  nvd            = 1;
  output         = 0;
  listsize       = 0;
  
  dof_values     = 0;
  constant       = 0.0;
  function       = 0;
  list           = 0;

  sprintf(name,"u");
  sprintf(label,"unknown field");
}
//-----------------------------------------------------------------------------
