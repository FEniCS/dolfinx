// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/Regulator.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/Partition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Partition::Partition(unsigned int N) : indices(N)
{
  // Get parameter for threshold
  threshold = dolfin_get("partitioning threshold");

  // Reset all indices
  for (unsigned int i = 0; i < N; i++)
    indices[i] = i;
}
//-----------------------------------------------------------------------------
Partition::~Partition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Partition::size() const
{
  return indices.size();
}
//-----------------------------------------------------------------------------
int Partition::index(unsigned int i) const
{
  dolfin_assert(i < indices.size());
  
  return indices[i];
}
//-----------------------------------------------------------------------------
void Partition::update(int offset, int& end, real& K, Adaptivity& adaptivity)
{
  // Compute time step for partition. We partition the components into two
  // groups, one group with k < K and one with k >= K.
  K = threshold * maximum(offset, adaptivity);

  // Comparison operator
  Less less(K, adaptivity);

  // Partition using std::partition
  NewArray<unsigned int>::iterator start = indices.begin();
  std::advance(start, offset);

  NewArray<unsigned int>::iterator middle =
    std::partition(start, indices.end(), less);

  // Compute pivot index
  end = std::distance(indices.begin(), middle);

  // Modify partition time step to the smallest k such that k >= K.
  K = minimum(offset, adaptivity, end);
}
//-----------------------------------------------------------------------------
void Partition::debug(unsigned int offset, unsigned int end, 
		      Adaptivity& adaptivity) const
{
  // This function can be used to debug the partitioning.
  
  cout << endl;
  for (unsigned int i = 0; i < indices.size(); i++)
  {
    if ( i == offset )
      cout << "--------------------------- offset" << endl;

    if ( i == end )
      cout << "--------------------------- end" << endl;

    cout << i << ": index = " << indices[i] 
	 << " k = " << adaptivity.regulator(indices[i]).timestep() << endl;
  }
  cout << endl;
}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset, Adaptivity& adaptivity) const
{
  real K = 0.0;

  for (unsigned int i = offset; i < indices.size(); i++)
    K = max(adaptivity.regulator(indices[i]).timestep(), K);

  return K;
}
//-----------------------------------------------------------------------------
real Partition::minimum(int offset, Adaptivity& adaptivity, int end) const
{
  real k = adaptivity.regulator(indices[offset]).timestep();

  for (int i = offset + 1; i < end; i++)
    k = min(adaptivity.regulator(indices[i]).timestep(), k);

  return k;
}
//-----------------------------------------------------------------------------
Partition::Less::Less(real& K, Adaptivity& adaptivity) : 
  K(K), adaptivity(adaptivity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Partition::Less::operator()(unsigned int index) const
{
  return adaptivity.regulator(index).timestep() >= K;
}
//-----------------------------------------------------------------------------
