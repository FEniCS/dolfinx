// Copyright (C) 2004 Johan Hoffman, Johan Jansson  and Anders Logg.
// Licensed under the GNU GPL Version 2.
//

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewVector::NewVector()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the vector
  v = 0;
}
//-----------------------------------------------------------------------------
NewVector::NewVector(int size)
{
  if(size < 0)
    dolfin_error("Size of vector must be non-negative.");

  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  v = 0;
  init(static_cast<unsigned int>(size));
}
//-----------------------------------------------------------------------------
NewVector::~NewVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void NewVector::init(unsigned int size)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if(v)
  {
    if(n == size)
    {
      return;      
    }
  }
  else
  {
    clear();
  }

  VecCreate(PETSC_COMM_WORLD, &v);
  VecSetSizes(v, PETSC_DECIDE, size);
  VecSetFromOptions(v);
}
//-----------------------------------------------------------------------------
void NewVector::clear()
{
  if(v)
  {
    VecDestroy(v);
  }

  v = 0;
}
//-----------------------------------------------------------------------------
unsigned int NewVector::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
void NewVector::show() const
{
  VecView(v, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
