// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/Regulator.h>
#include <dolfin/TimeSlabData.h>
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
void Partition::update(int offset, int& end, real& K, TimeSlabData& data)
{
  // Compute the largest time step
  K = maximum(offset, data) / threshold;

  // Comparison operator
  Less less(K, data);

  // Partition using std::partition
  NewArray<unsigned int>::iterator middle =
    std::partition(indices.begin() + offset, indices.end(), less);

  // Compute pivot index
  end = middle - indices.begin();
}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset, TimeSlabData& data) const
{
  real K = 0.0;

  for (unsigned int i = offset; i < indices.size(); i++)
    K = max(data.regulator(indices[i]).timestep(), K);

  return K;
}
//-----------------------------------------------------------------------------
Partition::Less::Less(real& K, TimeSlabData& data) : K(K), data(data)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Partition::Less::operator()(unsigned int index) const
{
  return data.regulator(index).timestep() >= K;
}
//-----------------------------------------------------------------------------
