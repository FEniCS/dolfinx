// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
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
  return components[i].index;
}
//-----------------------------------------------------------------------------
void Partition::update(TimeSlabData& data, int offset)
{
  // Compute new preliminary time steps for all elements, starting at
  // position offset, and compute the maximum time step.

  for (int i = offset; i < components.size(); i++) {

    // Component index
    int index = components[i].index;

    real k;

    if(index == 0)
      k = 0.2;
    else
      k = 0.1;
    /*
      if(data.component(index).size() != 0)
      {
      // Get time step
      k = data.component(index).last().newTimeStep();
      }
      else
      {
      }
    */

    dolfin_debug2("Partition: k[%d]: %f", i, k);

    // Save time step
    components[i].timestep = k;

  }
}
//-----------------------------------------------------------------------------
void Partition::partition(int offset, int& end, real& K)
{
  // Compute K

  //K = maximum(offset) / 2.0;
  K = maximum(offset) / 1.0;
  
  for (int i = offset; i < components.size(); i++)
     dolfin_debug2("Partition: k[%d]: %f", i, components[i].timestep);

  lessComponents foo(K);

  std::vector<Component>::iterator middle =
    std::partition(components.begin(), components.end(), foo);

  end = middle - components.begin();

  dolfin_debug1("end: %d", end);

  for (int i = offset; i < components.size(); i++)
    dolfin_debug2("Partition: k[%d]: %f", i, components[i].timestep);

}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset)
{
  real K = 0.0;

  for (int i = offset; i < components.size(); i++)
    K = max(components[i].timestep, K);

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
bool Partition::Component::operator<(Component &a)
{
  return timestep < a.timestep;
}
//-----------------------------------------------------------------------------
