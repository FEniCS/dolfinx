// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/Partition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Partition::Partition(int N, real timestep) : components(N)
{
  for (int i = 0; i < N; i++) {
    components[i].index = i;
    components[i].timestep = timestep;
  }
  
  valid = false;
  threshold = 1.0;
}
//-----------------------------------------------------------------------------
Partition::~Partition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Partition::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
int Partition::index(unsigned int i) const
{
  dolfin_assert(i < components.size());
  
  return components[i].index;
}
//-----------------------------------------------------------------------------
void Partition::update(int offset, RHS& f, TimeSlabData& data)
{
  // Compute new preliminary time steps for all elements, starting at
  // position offset, and compute the maximum time step.

  // Don't compute new time steps if we don't need to
  if (valid)
    return;
  
  // Compute new time steps for all components starting at offset
  for (unsigned int i = offset; i < components.size(); i++)
  {
    // Get last element for the component
    Element& element = data.component(index(i)).last();

    // Compute residual
    real r = element.computeResidual(f);

    // Compute new time step
    real k = element.computeTimeStep(r);
    
    // Save time step
    components[i].timestep = k;
  } 
}
//-----------------------------------------------------------------------------
void Partition::partition(int offset, int& end, real& K)
{
  // Compute the largest time step
  K = maximum(offset) / threshold;

  // Comparison operator
  Less less(K);

  // Partition using std::partition
  std::vector<ComponentIndex>::iterator middle =
    std::partition(components.begin() + offset, components.end(), less);

  // Compute pivot index
  end = middle - components.begin();

  // Change the state
  valid = true;
}
//-----------------------------------------------------------------------------
void Partition::invalidate()
{
  valid = false;
}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset) const
{
  real K = 0.0;

  for (unsigned int i = offset; i < components.size(); i++)
    K = max(components[i].timestep, K);

  return K;
}
//-----------------------------------------------------------------------------
