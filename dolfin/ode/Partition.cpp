// Copyright (C) 2004-2008 Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2004
// Last changed: 2009-09-08

#include <algorithm>
#include <cmath>

#include <dolfin/common/real.h>
#include <dolfin/log/dolfin_log.h>
#include "MultiAdaptivity.h"
#include "Partition.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Partition::Partition(uint N) : indices(N), threshold(0)
{
  // Reset all indices
  for (uint i = 0; i < N; i++)
    indices[i] = i;
}
//-----------------------------------------------------------------------------
Partition::~Partition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint Partition::size() const
{
  return indices.size();
}
//-----------------------------------------------------------------------------
dolfin::uint Partition::index(uint i) const
{
  assert(i < indices.size());
  return indices[i];
}
//-----------------------------------------------------------------------------
real Partition::update(uint offset, uint& end, MultiAdaptivity& adaptivity,
		       real K)
{
  // Compute time steps for partition. We partition the components into two
  // groups, one group with k < Kpivot and one with k >= Kpivot.

  // Get parameter for threshold (only first time)
  if (threshold == 0.0)
  {
    threshold = adaptivity.ode.parameters["partitioning_threshold"].get_real();
  }

  // Compute time step for partitioning
  real Kpivot = threshold * maximum(offset, adaptivity);

  // Comparison operator
  Less less(Kpivot, adaptivity);

  // Partition using std::partition
  std::vector<uint>::iterator start = indices.begin();
  std::advance(start, offset);
  std::vector<uint>::iterator middle = std::partition(start, indices.end(), less);

  // Compute pivot index
  end = std::distance(indices.begin(), middle);

  // Modify time step to the smallest k such that k >= Kpivot.
  Kpivot = minimum(offset, end, adaptivity);

  // Modify time step so K is a multiple of Kpivot
  //Kpivot = K / ceil(K / Kpivot);

  /*
  for (uint i = offset; i < end ; i++)
  {
    uint index = indices[i];
    cout << "i = " << index << ": " << adaptivity.timestep(index)
	 << " --> " << Kpivot << endl;
  }
  */

  return Kpivot;
}
//-----------------------------------------------------------------------------
void Partition::debug(uint offset, uint end) const
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
real Partition::maximum(uint offset, MultiAdaptivity& adaptivity) const
{
  real K = 0.0;

  for (uint i = offset; i < indices.size(); i++)
    K = std::max(adaptivity.timestep(indices[i]), K);

  return K;
}
//-----------------------------------------------------------------------------
real Partition::minimum(uint offset, uint end,
			   MultiAdaptivity& adaptivity) const
{
  real k = adaptivity.timestep(indices[offset]);

  for (uint i = offset + 1; i < end; i++)
    k = std::min(adaptivity.timestep(indices[i]), k);

  return k;
}
//-----------------------------------------------------------------------------
Partition::Less::Less(real& K, MultiAdaptivity& adaptivity)
  : K(K), adaptivity(adaptivity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Partition::Less::operator()(uint index) const
{
  return adaptivity.timestep(index) >= K;
}
//-----------------------------------------------------------------------------
