// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewTimeSlab.h>

using namespace dolfin;

//#define TEST_NEW

#ifdef TEST_NEW

//-----------------------------------------------------------------------------
NewTimeSlab::NewTimeSlab(real t0, real t1) : t0(t0), t1(t1)
{
  jx = 0;
  je = 0;
  jl = 0;
  jd = 0;
  
  ei = 0;
  es = 0;
  ej = 0;
  
  lt = 0;
  lw = 0;
  
  sk = 0;
  sn = 0;
  sm = 0;
}
//-----------------------------------------------------------------------------
NewTimeSlab::~NewTimeSlab()
{
  if ( jx ) delete [] jx;
  if ( je ) delete [] je;
  if ( jl ) delete [] jl;
  if ( jd ) delete [] jd;
  
  if ( ei ) delete [] ei;
  if ( es ) delete [] es;
  if ( ej ) delete [] ej;
  
  if ( lt ) delete [] lt;
  if ( lw ) delete [] lw;
  
  if ( sk ) delete [] sk;
  if ( sn ) delete [] sn;
  if ( sm ) delete [] sm;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::build()
{
  // Compute data sizes
  uint nj(0), ne(0), nl(0), ns(0);
  computeSize(partition, adaptivity, nj, ne, nl, ns);

  // Allocate data
  allocDofs(ns);
  allocElements(ne);
  allocStages(nl);
  allocSubSlabs(ns);

  // Build time slab
  createTimeSlab(adaptivity, partition, 0);
}
//-----------------------------------------------------------------------------
real NewTimeSlab::starttime() const
{
  return t0;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::endtime() const
{
  return t1;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::length() const
{
  return t1 - t0;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createTimeSlab(Adaptivity& adaptivity,
				 Partition& partition, uint offset)
{
  uint end = 0;
  real K = 0.0;
  
  // Update partitition 
  partition.update(offset, end, K, adaptivity);
  
  // Adjust and set the size of this time slab 
  setsize(K, adaptivity);

  // Recursively create elements for components with small time steps
  real t = t0;
  while ( t < t1 )
    t = create(adaptivity, partition, offset, t, t1);
  
  // Create sub slab
  uint subslab = createSubSlab();

  // Create stages
  createStages();
  
  // Create elements in sub slab
  createElements(s);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createElements(Adaptivity& adaptivity,
				       Partition& partition,
				       int offset, int end)
  
{
  // Get length of this time slab
  real k = length();
  
  // Create elements
  for (uint i = offset; i < end; i++)
  {
    // Create element
    uint element = createElement(s, index);

    // Create dofs for element
    createDofs(element);
  }
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createDofs()
{
  dolfin_assert((alloc_j.next + n -1) < alloc_size);

  // Get next available position
  uint pos = alloc_j.next;
  alloc_j.next += n;

  // Create dofs
  for (uint stage = 0; stage < n; stage++)
  {
    jx[pos + stage] = 0.0;
    je[pos + stage] = element;
    jl[pos + stage] = i;
    jd[pos + stage] = ?????;
  }
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createElement(subslab, index)
{
  dolfin_assert(alloc_e.next < alloc_e.size);
  // Check for space
  if ( alloc_e.next == alloc_e.alloc )
    allocElements();

  // Get next available position
  uint pos = alloc_e.next++;

  // Create new element
  ei[pos] = index;
  es[pos] = subslab;
  ej[pos] = initial dof of e;

  // Create dofs for element
  createDofs(pos);
}
//-----------------------------------------------------------------------------
void NewTimeSlab::createStages(t0, k, method)
{
  // Check for space
  dolfin_assert((alloc_l.next + n - 1) < alloc_l.size);
  
  // Get next available position
  uint pos = alloc_e.next;
  alloc_e.next += n;
  
  // Create stages
  for (uint i = 0; i < n; i++)
  {
    lt[pos + i] = t0 + ...;
    lw[pos + i] = plocka ut vikten;
  }
}
//-----------------------------------------------------------------------------
uint NewTimeSlab::createSubSlab()
{
  // Check for space
  dolfin_assert(alloc_s.next < alloc_s.size);
    allocSubSlabs();

  // Get next available position
  uint pos = alloc_s.next++;

  // Create new sub slab
  sk[pos] = k;
  sn[pos] = number of stages;
  sm[pos] = method of sub slab;
  
  return pos;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocDofs(uint newsize)
{
  if ( newsize <= alloc_j.size ) return

  Alloc::alloc(&jx, alloc_j.size, newsize);
  Alloc::alloc(&je, alloc_j.size, newsize);
  Alloc::alloc(&jl, alloc_j.size, newsize);
  Alloc::alloc(&jd, alloc_j.size, newsize);

  alloc_j.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocElements(uint newsize)
{
  if ( newsize <= alloc_e.size ) return

  Alloc::alloc(&ei, alloc_e.size, newsize);
  Alloc::alloc(&es, alloc_e.size, newsize);
  Alloc::alloc(&ej, alloc_e.size, newsize);

  alloc_e.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocStages(uint newsize)
{
  if ( newsize <= alloc_l.size ) return

  Alloc::alloc(&lt, alloc_l.size, newsize);
  Alloc::alloc(&lw, alloc_l.size, newsize);

  alloc_l.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocSubSlabs()
{
  if ( newsize <= alloc_s.size ) return

  Alloc::alloc(&sk, alloc_s.size, newsize);
  Alloc::alloc(&sn, alloc_s.size, newsize);
  Alloc::alloc(&sm, alloc_s.size, newsize);

  alloc_s.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::computeSize(Adaptivity& adaptivity, Partition& partition,
			      uint offset,
			      uint& nj, uint& ne, uint& nl; uint& ns) const
{
  // Recursively compute data sizes using the same algorithm as
  // for the recursive creation of the time slab

  // Add contribution from all sub slabs
  real t = t0;
  while ( t < t1 )
    t = computeSize(adaptivity, partition, offset, nj, ne, nl, ns);

  // Add contribution from this sub slab
  nj += n*(end - offset);
  ne += end - offset;
  nl += n;
  ns += 1;
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

#endif
