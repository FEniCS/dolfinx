// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/NewTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewTimeSlab::NewTimeSlab(const ODE&ode, const Method& method) : 
  se(0), st(0), ej(0), ei(0), jx(0), jd(0), de(0),
  ode(ode), method(method), partition(ode.size()), _t0(0), _t1(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewTimeSlab::~NewTimeSlab()
{
  if ( se ) delete [] se;
  if ( st ) delete [] st;
  if ( ej ) delete [] ej;
  if ( ei ) delete [] ei;
  if ( jx ) delete [] jx;
  if ( jd ) delete [] jd;
  if ( de ) delete [] de;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::build(real t0, real t1, Adaptivity& adaptivity)
{
  // Allocate data
  allocData(t0, t1, adaptivity);

  // Build time slab
  t1 = createTimeSlab(t0, t1, adaptivity, 0);

  // Save start and end time
  _t0 = t0;
  _t1 = t1;

  return t1;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::starttime() const
{
  return _t0;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::endtime() const
{
  return _t1;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::length() const
{
  return _t1 - _t0;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::createTimeSlab(real t0, real t1, Adaptivity& adaptivity,
				 uint offset)
{
  // Compute size of this sub slab
  uint end = 0;
  t1 = computeEndTime(t0, t1, adaptivity, offset, end);
  
  // Recursively create sub slabs for components with small time steps
  real t = t0;
  while ( t < t1 )
    t = createTimeSlab(t, t1, adaptivity, offset);
  
  // Create sub slab
  createSubSlab(t0, offset, end);

  return t1;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createSubSlab(real t0, uint offset, uint end)
{
  dolfin_assert(alloc_s.next < alloc_s.size);
  
  // Get next available position
  uint pos = alloc_s.next++;

  // Create new sub slab
  se[pos] = alloc_e.next;
  st[pos] = t0;

  // Create elements for sub slab
  for (uint i = offset; i < end; i++)
    createElement(partition.index(i));
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createElement(uint index)
{
  dolfin_assert(alloc_e.next < alloc_e.size);

  // Get next available position
  uint pos = alloc_e.next++;

  // Create new element
  ej[pos] = alloc_j.next;
  ei[pos] = index;

  // Create dofs for element
  createDofs(index);
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createDofs(uint index)
{
  dolfin_assert((alloc_j.next + method.size() - 1) < alloc_j.size);

  // Get next available position
  uint pos = alloc_j.next;
  alloc_j.next += method.size();

  // Create dofs
  for (uint stage = 0; stage < method.size(); stage++)
  {
    jx[pos + stage] = 0.0; // FIXME: Get initial value
    jd[pos + stage] = alloc_d.next;
    
    // Create dependencies for dof
    // FIXME: Should create dependencies here
  }
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createDeps()
{
  // FIXME: Not implemented
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocData(real t0, real t1, Adaptivity& adaptivity)
{
  // Compute size of s, e, and j mappings recursively
  uint ns(0), ne(0), nj(0);
  computeDataSize(ns, ne, nj, t0, t1, adaptivity, 0);
  
  // Compute size of d mapping (number of dependencies)
  uint nd(0);
  // FIXME: not implemented

  // Allocate data
  allocSubSlabs(ns);
  allocElements(ne);
  allocDofs(nj);
  allocDeps(nd);
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocSubSlabs(uint newsize)
{
  if ( newsize <= alloc_s.size ) return;

  Alloc::realloc(&se, alloc_s.size, newsize);
  Alloc::realloc(&st, alloc_s.size, newsize);

  alloc_s.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocElements(uint newsize)
{
  if ( newsize <= alloc_e.size ) return;

  Alloc::realloc(&ej, alloc_e.size, newsize);
  Alloc::realloc(&ei, alloc_e.size, newsize);

  alloc_e.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocDofs(uint newsize)
{
  if ( newsize <= alloc_j.size ) return;

  Alloc::realloc(&jx, alloc_j.size, newsize);
  Alloc::realloc(&jd, alloc_j.size, newsize);

  alloc_j.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocDeps(uint newsize)
{
  if ( newsize <= alloc_d.size ) return;

  Alloc::realloc(&de, alloc_d.size, newsize);

  alloc_d.size = newsize;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::computeEndTime(real t0, real t1, Adaptivity& adaptivity,
				 uint offset, uint& end)
{
  // Update partitition 
  real K = partition.update(offset, end, adaptivity);
  
  // Modify time step if we're close to the end time
  if ( K < adaptivity.threshold() * (t1 - t0) )
    t1 = t0 + K;
  
  return t1;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::computeDataSize(uint& ns, uint& ne, uint& nj,
				  real t0, real t1,
				  Adaptivity& adaptivity, uint offset)
{
  // Recursively compute data sizes using the same algorithm as
  // for the recursive creation of the time slab

  // Compute end time of this sub slab
  uint end = 0;
  t1 = computeEndTime(t0, t1, adaptivity, offset, end);
  
  // Add contribution from all sub slabs
  real t = t0;
  while ( t < t1 )
    t = computeDataSize(ns, ne, nj, t, t1, adaptivity, offset);

  // Add contribution from this sub slab
  ns += 1;
  ne += end - offset;
  nj += method.size()*(end - offset);

  return t1;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const NewTimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between t0 = " << timeslab.starttime()
	 << " and t1 = " << timeslab.endtime() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
