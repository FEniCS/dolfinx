// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GLOBALFIELD_HH
#define __GLOBALFIELD_HH

#include <dolfin/constants.h>
#include <dolfin/Grid.h>
#include "Output.h"

enum Representation { NONE, NODAL, CONSTANT, FUNCTION, LIST };
 
namespace dolfin {

  class Vector;
  
  class GlobalField {
  public:
	 
	 GlobalField(Grid *grid, real constant);
	 GlobalField(Grid *grid, Vector *x, int nvd = 1);
	 GlobalField(Grid *grid, const char *field);
	 GlobalField(GlobalField *f1, GlobalField *f2);
	 GlobalField(GlobalField *f1, GlobalField *f2, GlobalField *f3);
	 GlobalField(GlobalField **f, int n);
	 
	 ~GlobalField();
	 
	 void SetSize  (int no_data, ...);
	 void SetLabel (const char *name, const char *label, int i = 0);
	 
	 int  GetNoDof     ();
	 int  GetVectorDim ();
	 //int  GetDim       (int cell);
	 void GetLocalDof  (Cell *cell, real t, real *local_dofs, int component = 0);
	 
	 void Save();
	 void Save(real t);
	 
  private:
	 
	 void InitCommon(Grid *grid);
	 
	 Grid* grid;
	 int no_dof;
	 int nvd;
	 int listsize;
	 Representation representation;
	 
	 // Name and label of variable
	 char name[DOLFIN_LINELENGTH];
	 char label[DOLFIN_LINELENGTH];
	 
	 // Output
	 Output *output;
	 
	 // Different representations
	 Vector *dof_values;
	 real constant;    
	 real (*function)(real x, real y, real z, real t);
	 GlobalField **list;
	 
  };

}

#endif
