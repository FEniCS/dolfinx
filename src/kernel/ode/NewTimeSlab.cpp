// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/ODE.h>
#include <dolfin/Dependencies.h>
#include <dolfin/NewcGqMethod.h>
#include <dolfin/NewdGqMethod.h>
#include <dolfin/FixedPointSolver.h>
#include <dolfin/NewTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewTimeSlab::NewTimeSlab(ODE& ode) : 
  sa(0), sb(0), ei(0), es(0), ee(0), ed(0), jx(0), de(0),
  ns(0), ne(0), nj(0), nd(0), N(ode.size()), _a(0), _b(0),
  ode(ode), method(0), solver(0), adaptivity(ode), partition(N),
  elast(N), u0(N), u(0), emax(0)
{
  cout << "Experimental time slab: creating" << endl;
  cout << "  N = " << ode.size() << endl;

  // Choose method
  std::string m = dolfin_get("method");
  int q = dolfin_get("order");
  if ( m == "cg" )
    method = new NewcGqMethod(q);
  else
    method = new NewdGqMethod(q);

  // Choose solver
  solver = new FixedPointSolver(*this, *method);

  // Get initial data
  for (uint i = 0; i < N; i++)
    u0[i] = ode.u0(i);
  
  // Initialize solution vector
  u = new real[N];

  // Initialize transpose of dependencies if necessary
  dolfin_info("Computing transpose (inverse) of dependency pattern.");
  if ( ode.dependencies.sparse() && !ode.transpose.sparse() )
    ode.transpose.transp(ode.dependencies);
}
//-----------------------------------------------------------------------------
NewTimeSlab::~NewTimeSlab()
{
  if ( sa ) delete [] sa;
  if ( sb ) delete [] sb;
  if ( ei ) delete [] ei;
  if ( es ) delete [] es;
  if ( ee ) delete [] ee;
  if ( ed ) delete [] ed;
  if ( jx ) delete [] jx;
  if ( de ) delete [] de;

  if ( method ) delete method;
  if ( solver ) delete solver;

  if ( u ) delete [] u;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::build(real a, real b)
{
  cout << "Experimental time slab: building between " << a << " and " << b << endl;

  // Allocate data
  allocData(a, b);

  // Reset elast
  elast = -1;

  // Create time slab recursively
  b = createTimeSlab(a, b, 0);

  //cout << "de = "; Alloc::disp(de, nd);
  
  // Save start and end time
  _a = a;
  _b = b;

  cout << "Experimental time slab: finished building between " << a << " and " << b << endl;

  return b;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::solve()
{
  cout << "Experimental time slab: solving" << endl;

  solver->solve();

  cout << "Experimental time slab: system solved" << endl;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::shift()
{
  // Cover end time
  cover(_b);
  
  // Update the solution vector at the end time for each component
  for (uint i = 0; i < N; i++)
  {
    // Get last element of component
    const int e = elast[i];
    dolfin_assert(e != -1);
    dolfin_assert(sb[es[e]] == _b);
    
    // Get end-time value of component
    const int j = e * method->nsize();
    u[i] = jx[j + method->nsize() - 1];
  }

  // Compute residual and new time step for each component
  for (uint i = 0; i < N; i++)
  {
    // Get last element of component
    const int e = elast[i];
    dolfin_assert(e != -1);
    
    // Get element data
    const uint s = es[e];
    const uint j = e * method->nsize();
    const real a = sa[s];
    const real b = sb[s];
    const real k = b - a;
    
    // Get initial value for element (only necessary for cG)
    const int ep = ee[e];
    const uint jp = ep * method->nsize();
    const real x0 = ( ep != -1 ? jx[jp + method->nsize() - 1] : u0[i] );
    
    // Evaluate right-hand side at end-point (u is already updated)
    const real f = ode.f(u, b, i);

    // Compute residual
    const real r = method->residual(x0, jx + j, f, k);

    // Update adaptivity
    adaptivity.update(i, r, *method);
  }

  // Set initial value to end-time value (needs to be done last)
  for (uint i = 0; i < N; i++)
    u0[i] = u[i];
}
//-----------------------------------------------------------------------------
void NewTimeSlab::sample(real t)
{
  // Cover the given time
  cover(t);

  //cout << "t = " << t << " elast: ";
  //for (uint i = 0; i < N; i++)
  //  cout << elast[i] << " ";
  //cout << endl;
}
//-----------------------------------------------------------------------------
dolfin::uint NewTimeSlab::size() const
{
  return N;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::starttime() const
{
  return _a;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::endtime() const
{
  return _b;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::length() const
{
  return _b - _a;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::usample(uint i, real t)
{
  // Get element
  const int e = elast[i];
  dolfin_assert(e != -1);

  // Get element data
  const uint s = es[e];
  const uint j = e * method->nsize();
  const real a = sa[s];
  const real b = sb[s];
  const real k = b - a;

  // Get initial value for element (only necessary for cG)
  const int ep = ee[e];
  const uint jp = ep * method->nsize();
  const real x0 = ( ep != -1 ? jx[jp + method->nsize() - 1] : u0[i] );
  
  // Evaluate solution
  const real tau = (t - a) / k;
  const real value = method->ueval(x0, jx + j, tau);

  return value;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::ksample(uint i, real t)
{
  // Get element
  const int e = elast[i];
  dolfin_assert(e != -1);

  // Get element data
  const uint s = es[e];
  const real a = sa[s];
  const real b = sb[s];

  // Compute time step
  const real k = b - a;

  return k;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::rsample(uint i, real t)
{
  // Step to the correct element
  //uint e = cover(i, t);
  
  //dolfin_error("Not implemented");

  return 0.0;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::disp() const
{
  cout << "--------------------------------------------------------" << endl;

  dolfin_info("s: size = %d alloc = %d", ns, size_s.size);
  dolfin_info("e: size = %d alloc = %d", ne, size_e.size);
  dolfin_info("j: size = %d alloc = %d", nj, size_j.size);
  dolfin_info("d: size = %d alloc = %d", nd, size_d.size);

  cout << "sa = "; Alloc::disp(sa, ns);
  cout << "sb = "; Alloc::disp(sb, ns);
 
  cout << endl;

  cout << "ei = "; Alloc::disp(ei, ne);
  cout << "es = "; Alloc::disp(es, ne);  
  cout << "ee = "; Alloc::disp(ee, ne);
  cout << "ed = "; Alloc::disp(ed, ne);

  cout << endl;

  cout << "jx = "; Alloc::disp(jx, nj);

  cout << endl;

  cout << "de = "; Alloc::disp(de, nd);
}
//-----------------------------------------------------------------------------
void NewTimeSlab::allocData(real a, real b)
{ 
  // Use u to keep track of the latest time value for each component here
  for (uint i = 0; i < N; i++)
    u[i] = a;
  
  // Recursively compute data size
  ns = ne = nj = nd = 0;
  computeDataSize(a, b, 0);

  // Allocate data
  alloc_s(ns);
  alloc_e(ne);
  alloc_j(nj);
  alloc_d(nd);

  // Reset mapping de
  for (uint d = 0; d < nd; d++)
    de[d] = -1;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::createTimeSlab(real a, real b, uint offset)
{
  // Compute end time of this sub slab
  uint end = 0;
  b = computeEndTime(a, b, offset, end);

  //cout << "Creating sub slab between a = " << a << " and b = " << b << endl;

  // Create sub slab
  create_s(a, b, offset, end);

  // Recursively create sub slabs for components with small time steps
  real t = a;
  while ( t < b && end < partition.size() )
    t = createTimeSlab(t, b, end);
  
  return b;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::create_s(real a0, real b0, uint offset, uint end)
{
  dolfin_assert(size_s.next < size_s.size);
  
  // Get next available position
  uint pos = size_s.next++;

  // Create new sub slab
  sa[pos] = a0;
  sb[pos] = b0;

  // Create elements for sub slab
  for (uint n = offset; n < end; n++)
    create_e(partition.index(n), pos, a0, b0);

  // Create mapping ed
  for (uint n = offset; n < end; n++)
  {
    const uint index = partition.index(n);
    const int element = elast[index];
    dolfin_assert(element != -1);

    // Count number of dependencies from element
    size_d.next += countDependencies(index, b0);

    // Update mapping ed
    if ( element == 0 )
      ed[element] = 0;
    if ( element < static_cast<int>(ne - 1) )
      ed[element + 1] = size_d.next;
  }
}
//-----------------------------------------------------------------------------
void NewTimeSlab::create_e(uint index, uint subslab, real a, real b)
{
  dolfin_assert(size_e.next < size_e.size);
  
  // Get next available position
  uint pos = size_e.next++;

  //dolfin_info("  Creating element e = %d for i = %d at [%f, %f]", pos, index, a, b);
  
  // Create new element
  ei[pos] = index;
  es[pos] = subslab;
  ee[pos] = elast[index];

  // Create dofs for element
  create_j(index);

  // Create dependencies to element
  create_d(index, pos, subslab, a, b);

  // Update elast for component
  elast[index] = pos;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::create_j(uint index)
{
  dolfin_assert((size_j.next + method->nsize() - 1) < size_j.size);

  // Get next available position
  uint pos = size_j.next;
  size_j.next += method->nsize();

  // Create dofs
  for (uint stage = 0; stage < method->nsize(); stage++)
    jx[pos + stage] = u0[index];
}
//-----------------------------------------------------------------------------
void NewTimeSlab::create_d(uint i0, uint e0, uint s0, real a0, real b0)
{
  // Add dependencies to elements that depend on the given element if the
  // depending elements use larger time steps
  
  //dolfin_info("Checking dependencies to element %d (component %d)", element, index);

  // Get list of components depending on current component
  const NewArray<uint>& deps = ode.transpose[i0];
  
  // Iterate over dependencies
  for (uint pos = 0; pos < deps.size(); pos++)
  {
    // Get component index of other component
    const uint i1 = deps[pos];
        
    // Get other element
    const int e1 = elast[i1];
    
    //cout << "  Other element: " << e1 << endl;
    
    // Skip elements which have not been created (use smaller time steps)
    if ( e1 == -1 )
      continue;
    
    // Get data of other element
    const uint s1 = es[e1];
    const real a1 = sa[s1];
    const real b1 = sb[s1];
    const real k1 = b1 - a1;
    
    // Only add dependencies from components with larger time steps
    if ( !within(a0, b0, a1, b1) || s0 == s1 )
      continue;
    
    //dolfin_info("  Checking element %d (component %d)", e1, i1);
    
    // Iterate over dofs for element
    for (uint stage = 0; stage < method->nsize(); stage++)
    {
      //const uint j = j1 + stage;
      const real t = a1 + k1*method->npoint(stage);
      
      //dolfin_info("    Checking dof at t = %f", t);
      
      // Check if dof is contained in the current element
      if ( within(t, a0, b0) )
      {
	// Search for an empty position
	bool found = false;
	
	//cout << "    --- Creating dependency to element = " << element << endl;
	//cout << "    --- Starting at ed = " << ed[e1] << endl;
	//cout << "    de = "; Alloc::disp(ed, ne);
	//cout << "    nd = "; Alloc::disp(de, nd);	

	for (uint d = ed[e1]; d < ed[e1 + 1]; d++)
	{
	  if ( de[d] == -1 )
	  {
	    de[d] = e0;
	    found = true;
	    break;
	  }
	}
	dolfin_assert(found);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void NewTimeSlab::alloc_s(uint newsize)
{
  size_s.next = 0;

  if ( newsize <= size_s.size ) return;

  dolfin_info("Reallocating: ns = %d", newsize);

  Alloc::realloc(&sa, size_s.size, newsize);
  Alloc::realloc(&sb, size_s.size, newsize);

  size_s.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::alloc_e(uint newsize)
{
  size_e.next = 0;

  if ( newsize <= size_e.size ) return;

  dolfin_info("Reallocating: ne = %d", newsize);

  Alloc::realloc(&ei, size_e.size, newsize);
  Alloc::realloc(&es, size_e.size, newsize);
  Alloc::realloc(&ee, size_e.size, newsize);
  Alloc::realloc(&ed, size_e.size, newsize);

  size_e.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::alloc_j(uint newsize)
{
  size_j.next = 0;

  if ( newsize <= size_j.size ) return;

  dolfin_info("Reallocating: nj = %d", newsize);

  Alloc::realloc(&jx, size_j.size, newsize);

  size_j.size = newsize;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::alloc_d(uint newsize)
{
  size_d.next = 0;

  if ( newsize <= size_d.size ) return;

  dolfin_info("Reallocating: nd = %d", newsize);

  Alloc::realloc(&de, size_d.size, newsize);

  size_d.size = newsize;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::computeEndTime(real a, real b, uint offset, uint& end)
{
  // Update partitition 
  real K = partition.update(offset, end, adaptivity);

  //partition.debug(offset, end);
  
  // Modify time step if we're close to the end time
  if ( K < adaptivity.threshold() * (b - a) )
    b = a + K;
  
  return b;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::computeDataSize(real a, real b, uint offset)
{
  // Recursively compute data sizes using the same algorithm as
  // for the recursive creation of the time slab

  // Compute end time of this sub slab
  uint end = 0;
  b = computeEndTime(a, b, offset, end);

  // Use u to keep track of the latest time value for each component here
  for (uint n = offset; n < end; n++)
    u[partition.index(n)] = b;

  // Add contribution from this sub slab
  ns += 1;
  ne += end - offset;
  nj += method->nsize()*(end - offset);
  for (uint n = offset; n < end; n++)
    nd += countDependencies(partition.index(n));

  // Add contribution from all sub slabs
  real t = a;
  while ( t < b && end < partition.size() )
    t = computeDataSize(t, b, end);

  return b;
}
//-----------------------------------------------------------------------------
dolfin::uint NewTimeSlab::countDependencies(uint i0)
{
  // Count the number of dependencies to components with smaller time steps
  // for the given component. This version is used before any elements are
  // created when we recursively compute the data size. We then use the
  // array u to store the latest time value for each component.

  uint n = 0;

  // Get list of dependencies for current component index
  const NewArray<uint>& deps = ode.dependencies[i0];
  
  // Iterate over dependencies
  for (uint pos = 0; pos < deps.size(); pos++)
  {
    // Get index of other component
    const uint i1 = deps[pos];
    
    // Use u to keep track of the latest time value for each component here
    if ( u[i0] > (u[i1] + DOLFIN_EPS) )
      n += method->nsize();
  }
  
  return n;
}
//-----------------------------------------------------------------------------
dolfin::uint NewTimeSlab::countDependencies(uint i0, real b0)
{
  // Count the number of dependencies to components with smaller time steps
  // for the given component. This version is used at the time of creation
  // of elements and we may then get the time values of already created
  // elements.

  uint n = 0;

  // Get list of dependencies for current component index
  const NewArray<uint>& deps = ode.dependencies[i0];
  
  // Iterate over dependencies
  for (uint pos = 0; pos < deps.size(); pos++)
  {
    // Get index of other component
    const uint i1 = deps[pos];
    
    // Get last element for component
    const int e1 = elast[i1];
    
    // If we have not yet created the element, then it has not reached b0
    if ( e1 == -1 )
    {
      n += method->nsize();
      continue;
    }
    
    // Need to check end time value of element
    const uint s1 = es[e1];
    const real b1 = sb[s1];
    
    // Check if the component has reached b0
    if ( b1 < (b0 - DOLFIN_EPS) )
    {
      n += method->nsize();
    }
  }

  return n;
}
//-----------------------------------------------------------------------------
bool NewTimeSlab::within(real t, real a, real b) const
{
  // Check if time is within the given interval, choosing the left interval
  // if we are close to the edge

  return (a + DOLFIN_EPS) < t && t <= (b + DOLFIN_EPS);
}
//-----------------------------------------------------------------------------
bool NewTimeSlab::within(real a0, real b0, real a1, real b1) const
{
  // Check if [a0, b0] is contained in [a1, b1]

  return a1 <= (a0 + DOLFIN_EPS) && (b0 - DOLFIN_EPS) <= b1;
}
//-----------------------------------------------------------------------------
dolfin::uint NewTimeSlab::cover(int subslab, uint element)
{
  // Check if we are still on the same sub slab
  if ( subslab == static_cast<int>(es[element]) )
    return subslab;

  // Get next sub slab
  subslab = es[element];
  
  // Update elast for all elements in the sub slab
  for (uint e = element; e < ne; e++)
  {
    // Check if we have reached the next sub slab
    if ( static_cast<int>(es[e]) != subslab )
      break;

    // Update elast 
    elast[ei[e]] = e;
  }

  return subslab;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::cover(real t)
{
  //cout << "Covering t = " << t << endl;

  // Check if t is covered for all components
  bool ok = true;
  for (uint i = 0; i < N; i++)
  {
    // Get last covered element for the component
    const int e = elast[i];

    // Check if we need to start from the beginning
    if ( e == -1 )
    {
      //cout << "Need to start from the beginning since e = -1" << endl;
      emax = 0;
      ok = false;
      break;
    }

    // Get element data
    const uint s = es[e];
    const real a = sa[s];
    const real b = sb[s];

    // Check if we need to start from the beginning
    if ( t < (a + DOLFIN_EPS) )
    {
      //cout << "Need to start from the beginning since we have stepped to far" << endl;
      emax = 0;
      ok = false;
      break;
    }

    // Check if we need to search forward, starting at e = emax
    if ( t > (b + DOLFIN_EPS) )
    {
      //cout << "Need to search forward" << endl;
      ok = false;
      break;
    }
  }

  // If ok is true, then a + DOLFIN_EPS <= t <= b + DOLFIN_EPS for all components
  if ( ok )
    return;

  // Reset sampling if necessary
  if ( emax >= ne )
    emax = 0;
  else
  {
    const uint s = es[emax];
    const real a = sa[s];
    
    if ( t < (a + DOLFIN_EPS) )
      emax = 0;
  }

  //cout << "Starting search at e = " << emax << endl;

  // Iterate forward until t is covered for all components
  for (uint e = emax; e < ne; e++)
  {
    // Get element data
    const uint s = es[e];
    const uint i = ei[e];
    const real a = sa[s];

    //cout << "  Checking element:" << endl;
    //cout << "    e = " << e << endl;
    //cout << "    i = " << i << endl;
    //cout << "    a = " << a << endl;
    //cout << "    t = " << t << endl;

    // Check if we have stepped far enough
    if ( t < (a + DOLFIN_EPS) && _a < (a - DOLFIN_EPS) )
      break;

    //cout << "  Element covered" << endl;

    // Cover element
    elast[i] = e;
    emax = e;
  }

  //cout << "Time covered" << endl << endl;
}
//-----------------------------------------------------------------------------
void NewTimeSlab::feval(real* f, uint s0, uint e0, uint i0, 
			real a0, real b0, real k0)
{
  //cout << "  Evaluating f for element " << e0
  //     << ": i = " << i0 << " a0 = " << a0 << " b0 = " << b0 << endl;

  // Get list of dependencies for given component index
  const NewArray<uint>& deps = ode.dependencies[i0];

  // Get first dependency to components with smaller time steps for element
  uint d = ed[e0];

  // Compute number of such dependencies for each nodal point
  const uint end = ( e0 < (ne - 1) ? ed[e0 + 1] : nd );
  const uint ndep = (end - d) / method->nsize();
  dolfin_assert(ndep * method->nsize() == (end - d));

  // Evaluate the right-hand side at all quadrature points
  for (uint n = 0; n < method->qsize(); n++)
  {
    // Compute quadrature point
    const real t = a0 + k0*method->qpoint(n);
    //cout << "  Evaluating u at t = " << t << endl;

    //cout << "  Updating for large time steps" << endl;

    // Update values for components with larger and equal time steps,
    // also including the initial value from components with small
    // time steps (needed for cG)
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      // Get element
      const uint i1 = deps[pos];
      const int e1 = elast[i1];

      // Special case, component has no latest element
      if ( e1 == -1 )
      {
	if ( t < (a0 + DOLFIN_EPS) )
	  u[i1] = u0[i1];
	continue;
      }

      // Get element data
      const uint s1 = es[e1];
      const real b1 = sb[s1];

      // Skip components with smaller time steps
      if ( b1 < (t - DOLFIN_EPS) )
       	continue;
      
      //cout << "    i1 = " << i1 << " e1 = " << e1 << endl;
      
      // Get initial value for element (only necessary for cG)
      const int ep = ee[e1];
      const uint jp = ep * method->nsize();
      const real x0 = ( ep != -1 ? jx[jp + method->nsize() - 1] : u0[i1] );

      // Use fast evaluation for elements in the same sub slab
      const uint j1 = e1 * method->nsize();
      if ( s0 == s1 )
	u[i1] = method->ueval(x0, jx + j1, n);
      else
      {
	const real a1 = sa[s1];
	const real k1 = b1 - a1;
	const real tau = (t - a1) / k1;
	u[i1] = method->ueval(x0, jx + j1, tau);
      }
    }

    //cout << "  Updating for small time steps" << endl;

    // Update values for components with smaller time steps, not including
    // the initial value (left end-point value). This is handled above
    // in the update for components with large time steps
    if ( t > (a0 + DOLFIN_EPS) )
    {
      for (uint dep = 0; dep < ndep; dep++)
      {
	// Get element
	const int e1 = de[d++];
	dolfin_assert(e1 != -1);

	// Get element data
	const uint i1 = ei[e1];
	const uint s1 = es[e1];
	const real a1 = sa[s1];
	const real b1 = sb[s1];
	const real k1 = b1 - a1;
	
	//cout << "    i1 = " << i1 << " e1 = " << e1 << endl;

	// Get initial value for element (only necessary for cG)
	const int ep = ee[e1];
	const uint jp = ep * method->nsize();
	const real x0 = ( ep != -1 ? jx[jp + method->nsize() - 1] : u0[i1] );

	// Evaluate component
	const real tau = (t - a1) / k1;
	const uint j1 = e1 * method->nsize();
	u[i1] = method->ueval(x0, jx + j1, tau);
      }
    }
    
    // Evaluate right-hand side
    f[n] = ode.f(u, t, i0);
  }
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const NewTimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between a = " << timeslab.starttime()
	 << " and b = " << timeslab.endtime() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
