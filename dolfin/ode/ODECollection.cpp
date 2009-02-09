// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-09
// Last changed: 2009-02-09

#include "ODECollection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODECollection::ODECollection(uint n, uint N, real T) : ODE(N, T)
{
  message("Creating ODE collection of size %d x %d.", n, N);

  // Allocate state vectors
  states = new real[n*N];
}
//-----------------------------------------------------------------------------
ODECollection::~ODECollection()
{
  delete [] states;
}
//-----------------------------------------------------------------------------
void ODECollection::u0(real* u)
{
  cout << "Requesting initial data for ODE number x" << endl;
}
//-----------------------------------------------------------------------------
