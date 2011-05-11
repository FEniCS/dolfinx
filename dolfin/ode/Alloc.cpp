// Copyright (C) 2004-2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2004-12-21
// Last changed: 2009-08-10

#include <dolfin/log/dolfin_log.h>
#include "Alloc.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Alloc::Alloc() : size(0), next(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Alloc::realloc(int** data, uint oldsize, uint newsize)
{
  assert(newsize > oldsize);

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
  assert(newsize > oldsize);

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
  assert(newsize > oldsize);

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
void Alloc::display(uint* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
void Alloc::display(int* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
void Alloc::display(real* data, uint size)
{
  cout << "[ ";
  for (uint i = 0; i < size; i++)
    cout << data[i] << " ";
  cout << "]" << endl;
}
//-----------------------------------------------------------------------------
