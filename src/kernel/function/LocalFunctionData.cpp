// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2005-11-28

#include <dolfin/FiniteElement.h>
#include <dolfin/Point.h>
#include <dolfin/LocalFunctionData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LocalFunctionData::LocalFunctionData() 
  : dofs(0), components(0), points(0), values(0), n(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LocalFunctionData::~LocalFunctionData()
{
  // Clear data if initialized
  clear();
}
//-----------------------------------------------------------------------------
void LocalFunctionData::init(const FiniteElement& element)
{
  // Check if data is already initialized correctly
  if ( n == element.spacedim() )
    return;

  // Clear data if initialized
  clear();

  // Initialize local degrees of freedom
  dofs = new int[element.spacedim()];

  // Initialize local components
  components = new uint[element.spacedim()];

  // Initialize local nodal points
  points = new Point[element.spacedim()];

  // Initialize local vertex values
  if ( element.rank() == 0 )
    values = new real[1];
  else
    values = new real[element.tensordim(0)];

  // Save dimension of local function space
  n = element.spacedim();
}
//-----------------------------------------------------------------------------
void LocalFunctionData::clear()
{
  if ( dofs )
    delete [] dofs;
  dofs = 0;
  
  if ( components )
    delete [] components;
  components = 0;
  
  if ( points )
    delete [] points;
  points = 0;
  
  if ( values )
    delete [] values;
  values = 0;
}
//-----------------------------------------------------------------------------
