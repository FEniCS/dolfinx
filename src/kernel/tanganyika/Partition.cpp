// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/Partition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Partition::Partition(int N, real timestep)
{
  components.init(N);

  for (int i = 0; i < N; i++) {
    components(i).index = i;
    components(i).timestep = timestep;
  }
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
int Partition::index(int i) const
{
  return components(i).index;
}
//-----------------------------------------------------------------------------
void Partition::update(TimeSlabData& data, int offset)
{
  // Compute new preliminary time steps for all elements, starting at
  // position offset, and compute the maximum time step.

  for (int i = offset; i < components.size(); i++) {

    // Component index
    int index = components(i).index;

    // Get time step
    real k = data.component(index).last().newTimeStep();

    // Save time step
    components(i).timestep = k;

  }
}
//-----------------------------------------------------------------------------
void Partition::partition(int offset, int& end, real& K)
{
  // Move all components with time steps larger than K to the top of
  // the list, at positions [offset ... end-1]. We move through the
  // list with two different indices and swap components. The
  // algorithm is O(components.size() - offset) and reminds about the
  // the partition used in quick sort.

  K = maximum(offset) / 2.0;
  int n = components.size();

  int i = offset;
  int j = offset;

  while ( true ) {
    
    // Move j to the next component with large time step
    for (; j < n; j++)
      if ( components(j).timestep > K )
	break;
    
    // If we couldn't find the component we are done
    if ( j == n )
      break;
    
    // Swap the components
    components.swap(i,j);
    
    // Move i to the next component
    i++;

  }

  end = i;
}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset)
{
  real K = 0.0;

  for (int i = offset; i < components.size(); i++)
    K = max(components(i).timestep, K);

  return K;
}
//-----------------------------------------------------------------------------
Partition::Component::Component()
{
  index = 0;
  timestep = 0.0;
}
//-----------------------------------------------------------------------------
Partition::Component::~Component()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Partition::Component::operator=(int zero)
{
  // This is needed for ShortList
  index = 0;
  timestep = 0.0;
}
//-----------------------------------------------------------------------------
