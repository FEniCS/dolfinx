// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-06
// Last changed: 2008-04-22

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/la/uBlasVector.h>
#include "ODE.h"
#include "Dependencies.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Dependencies::Dependencies(uint N) :
  N(N), increment(dolfin_get("ODE sparsity check increment")), _sparse(false)
{
  // Use dense dependency pattern by default
  ddep.reserve(N);
  ddep.resize(N);
  for (uint i = 0; i < N; i++)
    ddep[i] = i;
}
//-----------------------------------------------------------------------------
Dependencies::~Dependencies()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Dependencies::setsize(uint i, uint size)
{
  // Prepare sparse pattern if necessary
  makeSparse();
  
  // Set size of row
  sdep[i].reserve(size);
}
//-----------------------------------------------------------------------------
void Dependencies::set(uint i, uint j, bool checknew)
{
  // Prepare sparse pattern if necessary
  makeSparse();

  // Check if the size has been specified
  if ( sdep[i].size() == sdep[i].capacity() )
    warning("Specify number of dependencies first for better performance.");

  // Check if the dependency already exists before inserting it
  if ( checknew )
  {
    for (uint k = 0; k < sdep[i].size(); k++)
      if ( sdep[i][k] == j )
	return;
  }

  // Add the dependency
  sdep[i].push_back(j);
}
//-----------------------------------------------------------------------------
void Dependencies::set(const uBlasSparseMatrix& A)
{
  // Prepare sparse pattern if necessary
  makeSparse();

  // Check dimension of matrix
  if ( A.size(0) != N )
    error("Incorrect matrix dimensions for dependency pattern.");

  // Get data from matrix
  for (uint i = 0; i < N; i++)
  {
    // FIXME: Could add function to return sparsity pattern
    Array<int> columns;
    Array<real> values;
    int ncols = 0;
    A.getRow(i, ncols, columns, values); 
    setsize(i, ncols);
    for (uint j = 0; j < static_cast<uint>(ncols); j++)
      set(i, columns[j]);
  }
}
//-----------------------------------------------------------------------------
void Dependencies::transp(const Dependencies& dependencies)
{
  // Don't compute dependency pattern is full
  if ( !dependencies._sparse )
  {
    if ( _sparse )
    {
      for (uint i = 0; i < N; i++)
        sdep[i].clear();
      sdep.clear();
      
      _sparse = false;
    }
    
    return;
  }

  // Prepare sparse pattern if necessary
  makeSparse();

  // Count the number of dependencies
  Array<uint> rowsizes(N);
  rowsizes = 0;
  for (uint i = 0; i < N; i++)
  {
    const Array<uint>& row(dependencies.sdep[i]);
    for (uint pos = 0; pos < row.size(); ++pos)
      rowsizes[row[pos]]++;
  }

  // Set row sizes
  for (uint i = 0; i < N; i++)
    setsize(i, rowsizes[i]);

  // Set dependencies
  for (uint i = 0; i < N; i++)
  {
    const Array<uint>& row(dependencies.sdep[i]);
    for (uint pos = 0; pos < row.size(); ++pos)
      set(row[pos], i);
  }
}
//-----------------------------------------------------------------------------
void Dependencies::detect(ODE& ode)
{
  // Prepare sparse pattern if necessary
  makeSparse();

  // Randomize solution vector
  uBlasVector u(N);
  for (uint i = 0; i < N; i++)
    u[i] = rand();
  
  // Check dependencies for all components
  Progress p("Computing sparsity", N);
  uint sum = 0;
  for (uint i = 0; i < N; i++)
  {
    // Count the number of dependencies
    uint size = 0;
    real f0 = ode.f(u, 0.0, i);
    for (uint j = 0; j < N; j++)
      if ( checkDependency(ode, u, f0, i, j) )
        size++;
    
    // Compute total number of dependencies
    sum += size;
    
    // Set size of row
    setsize(i, size);

    // Set the dependencies
    for (uint j = 0; j < N; j++)
      if ( checkDependency(ode, u, f0, i, j) )
	set(i, j);
    
    // Update progress
    p = i;
  }

  message("Automatically detected %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
bool Dependencies::sparse() const
{
  return _sparse;
}
//-----------------------------------------------------------------------------
void Dependencies::disp() const
{
  if ( _sparse )
  {
    message("Dependency pattern:");
    for (uint i = 0; i < N; i++)
    {
      cout << i << ":";
      for (uint pos = 0; pos < sdep[i].size(); ++pos)
        cout << " " << sdep[i][pos];
      cout << endl;
    }
  }
  else  
    message("Dependency pattern: dense");
}
//-----------------------------------------------------------------------------
bool Dependencies::checkDependency(ODE& ode, uBlasVector& u, real f0,
				   uint i, uint j)
{
  // Save original value
  real uj = u[j];

  // Change value and compute new value for f_i
  u[j] += increment;
  real f = ode.f(u, 0.0, i);

  // Restore the value
  u[j] = uj;

  // Compare function values
  return fabs(f - f0) > DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
void Dependencies::makeSparse()
{
  if ( _sparse )
    return;
  
  sdep.reserve(N);
  sdep.resize(N);
  
  ddep.clear();
    
    _sparse = true;
}
//-----------------------------------------------------------------------------
