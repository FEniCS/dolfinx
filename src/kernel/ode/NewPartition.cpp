// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Regulator.h>
#include <dolfin/MultiAdaptivity.h>
#include <dolfin/NewPartition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPartition::NewPartition(uint N) : indices(N)
{
  // Get parameter for threshold
  threshold = dolfin_get("partitioning threshold");

  // Reset all indices
  for (uint i = 0; i < N; i++)
    indices[i] = i;
}
//-----------------------------------------------------------------------------
NewPartition::~NewPartition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint NewPartition::size() const
{
  return indices.size();
}
//-----------------------------------------------------------------------------
dolfin::uint NewPartition::index(uint i) const
{
  dolfin_assert(i < indices.size());  
  return indices[i];
}
//-----------------------------------------------------------------------------
real NewPartition::update(uint offset, uint& end, MultiAdaptivity& adaptivity)
{
  // Compute time steps for partition. We partition the components into two
  // groups, one group with k < K and one with k >= K.

  // Compute time step for partitioning
  real K = threshold * maximum(offset, adaptivity);

  // Comparison operator
  Less less(K, adaptivity);

  // NewPartition using std::partition
  NewArray<uint>::iterator start = indices.begin();
  std::advance(start, offset);
  NewArray<uint>::iterator middle = std::partition(start, indices.end(), less);
  
  // Compute pivot index
  end = std::distance(indices.begin(), middle);

  // Modify time step to the smallest k such that k >= K.
  K = minimum(offset, end, adaptivity);

  return K;
}
//-----------------------------------------------------------------------------
void NewPartition::debug(uint offset, uint end) const
{
  cout << "Partition:";

  for (uint i = 0; i < indices.size(); i++)
  {
    if ( i == offset || i == end )
      cout << " |";

    cout << " " << indices[i];
  }

  cout << endl;
}
//-----------------------------------------------------------------------------
real NewPartition::maximum(uint offset, MultiAdaptivity& adaptivity) const
{
  real K = 0.0;

  for (uint i = offset; i < indices.size(); i++)
    K = max(adaptivity.timestep(indices[i]), K);

  return K;
}
//-----------------------------------------------------------------------------
real NewPartition::minimum(uint offset, uint end,
			   MultiAdaptivity& adaptivity) const
{
  real k = adaptivity.timestep(indices[offset]);

  for (uint i = offset + 1; i < end; i++)
    k = min(adaptivity.timestep(indices[i]), k);

  return k;
}
//-----------------------------------------------------------------------------
NewPartition::Less::Less(real& K, MultiAdaptivity& adaptivity)
  : K(K), adaptivity(adaptivity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool NewPartition::Less::operator()(uint index) const
{
  return adaptivity.timestep(index) >= K;
}
//-----------------------------------------------------------------------------
