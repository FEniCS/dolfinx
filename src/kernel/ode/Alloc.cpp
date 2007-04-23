// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-12-21
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/Alloc.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Alloc::Alloc() : size(0), next(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Alloc::realloc(int** data, uint oldsize, uint newsize)
{
  dolfin_assert(newsize > oldsize);
  
  // Allocate new data
  int* newdata = new int[newsize];

  // Copy old data
  for (uint i = 0; i < oldsize; i++)
    newdata[i] = (*data)[i];

  // Set default values
  for (uint i = oldsize; i < newsize; i++)
    newdata[i] = 0;

  // Delete old data and replace
  if ( *data ) delete [] *data;
  *data = newdata;
}
//-----------------------------------------------------------------------------
void Alloc::realloc(uint** data, uint oldsize, uint newsize)
{
  dolfin_assert(newsize > oldsize);
  
  // Allocate new data
  uint* newdata = new uint[newsize];

  // Copy old data
  for (uint i = 0; i < oldsize; i++)
    newdata[i] = (*data)[i];

  // Set default values
  for (uint i = oldsize; i < newsize; i++)
    newdata[i] = 0;

  // Delete old data and replace
  if ( *data ) delete [] *data;
  *data = newdata;
}
//-----------------------------------------------------------------------------
void Alloc::realloc(real** data, uint oldsize, uint newsize)
{
  dolfin_assert(newsize > oldsize);

  // Allocate new data
  real* newdata = new real[newsize];

  // Copy old data
  for (uint i = 0; i < oldsize; i++)
    newdata[i] = (*data)[i];

  // Set default values
  for (uint i = oldsize; i < newsize; i++)
    newdata[i] = 0.0;

  // Delete old data and replace
  if ( *data ) delete [] *data;
  *data = newdata;
}
//-----------------------------------------------------------------------------
void Alloc::disp(uint* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
void Alloc::disp(int* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
void Alloc::disp(real* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
