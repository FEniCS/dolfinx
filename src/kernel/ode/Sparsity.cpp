// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Sparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Sparsity::Sparsity(unsigned int N) :
  N(N), increment(dolfin_get("sparsity check increment"))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Sparsity::~Sparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Sparsity::clear()
{
  if ( pattern.empty() )
    return;

  for (unsigned int i = 0; i < N; i++)
    pattern[i].clear();

  pattern.clear();
}
//-----------------------------------------------------------------------------
void Sparsity::clear(unsigned int i)
{
  if ( pattern.empty() )
    return;

  pattern[i].clear();
}
//-----------------------------------------------------------------------------
void Sparsity::setsize(unsigned int i, unsigned int size)
{
  // Initialize list
  if ( pattern.empty() )
  {
    pattern.reserve(N);
    pattern.resize(N);
  }
  
  // Set size of row
  pattern[i].reserve(size);
}
//-----------------------------------------------------------------------------
void Sparsity::set(unsigned int i, unsigned int j)
{
  if ( pattern.empty() )
    dolfin_error("Number of dependencies for given components has not been specified.");

  pattern[i].push_back(j);
}
//-----------------------------------------------------------------------------
void Sparsity::set(const Matrix& A)
{
  // Check dimension of matrix
  if ( !A.size(0) != N )
    dolfin_error("Incorrect matrix dimensions for sparsity pattern.");

  // Clear previous dependencies
  clear();

  // Set dependencies for each row
  for (unsigned int i = 0; i < N; i++)
  {
    setsize(i, A.rowsize(i));
    unsigned int j = 0;
    for (unsigned int pos = 0; !A.endrow(i, pos); ++pos)
    {
      if ( fabs(A(i, j, pos)) > DOLFIN_EPS )
	pattern[i].push_back(j);
    }
  }
}
//-----------------------------------------------------------------------------
void Sparsity::transp(const Sparsity& sparsity)
{
  // Clear previous dependencies
  clear();

  // Don't compute dependencies if sparsity pattern is full
  if ( !sparsity.sparse() )
    return;

  // Count the number of dependencies
  NewArray<unsigned int> rowsizes(N);
  rowsizes = 0;
  for (unsigned int i = 0; i < N; i++)
  {
    const NewArray<unsigned int>& row(sparsity.row(i));
    for (unsigned int pos = 0; pos < row.size(); ++pos)
      rowsizes[row[pos]]++;
  }

  // Set row sizes
  for (unsigned int i = 0; i < N; i++)
    setsize(i, rowsizes[i]);

  // Set dependencies
  for (unsigned int i = 0; i < N; i++)
  {
    const NewArray<unsigned int>& row(sparsity.row(i));
    for (unsigned int pos = 0; pos < row.size(); ++pos)
      set(row[pos], i);
  }
}
//-----------------------------------------------------------------------------
void Sparsity::detect(ODE& ode)
{
  // Clear previous dependencies
  clear();

  // Randomize solution vector
  Vector u(N);
  u.rand();
  
  // Check dependencies for all components
  Progress p("Computing sparsity", N);
  unsigned int sum = 0;
  for (unsigned int i = 0; i < N; i++)
  {
    // Count the number of dependencies
    unsigned int size = 0;
    real f0 = ode.f(u, 0.0, i);
    for (unsigned int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	size++;
    
    // Compute total number of dependencies
    sum += size;

    // Set size of row
    setsize(i, size);

    // Set the dependencies
    for (unsigned int j = 0; j < N; j++)
      if ( checkdep(ode, u, f0, i, j) )
	set(i, j);
    
    // Update progress
    p = i;
  }

  dolfin_info("Automatically detected %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
NewArray<unsigned int>& Sparsity::row(unsigned int i)
{
  dolfin_assert(sparse());
  return pattern[i];
}
//-----------------------------------------------------------------------------
const NewArray<unsigned int>& Sparsity::row(unsigned int i) const
{
  dolfin_assert(sparse());
  return pattern[i];
}
//-----------------------------------------------------------------------------
void Sparsity::show() const
{
  dolfin_info("Sparsity pattern:");
  
  if ( !sparse() )
    dolfin_info("Unspecified / full");
  else
  {
    for (unsigned int i = 0; i < N; i++)
    {
      cout << i << ":";
      for (unsigned int pos = 0; pos < pattern[i].size(); ++pos)
	cout << " " << pattern[i][pos];
      cout << endl;
    }
  }
}
//-----------------------------------------------------------------------------
bool Sparsity::checkdep(ODE& ode, Vector& u, real f0,
			unsigned int i, unsigned int j)
{
  // Save original value
  real uj = u(j);

  // Change value and compute new value for f_i
  u(j) += increment;
  real f = ode.f(u, 0.0, i);

  // Restore the value
  u(j) = uj;

  // Compare function values
  return fabs(f - f0) > DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
