// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/Regulator.h>
#include <dolfin/TimeSteppingData.h>
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
void Partition::update(int offset, int& end, real& K, TimeSteppingData& data)
{
  // Compute the largest time step
  K = threshold * maximum(offset, data);

  // Comparison operator
  Less less(K, data);

  // Partition using std::partition
  NewArray<unsigned int>::iterator middle =
    std::partition(indices.begin() + offset, indices.end(), less);

  // Compute pivot index
  end = middle - indices.begin();
}
//-----------------------------------------------------------------------------
void Partition::debug(unsigned int offset, unsigned int end, 
		      TimeSteppingData& data) const
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
	 << " k = " << data.regulator(indices[i]).timestep() << endl;
  }
  cout << endl;
}
//-----------------------------------------------------------------------------
real Partition::maximum(int offset, TimeSteppingData& data) const
{
  real K = 0.0;

  for (unsigned int i = offset; i < indices.size(); i++)
    K = max(data.regulator(indices[i]).timestep(), K);

  return K;
}
//-----------------------------------------------------------------------------
Partition::Less::Less(real& K, TimeSteppingData& data) : K(K), data(data)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Partition::Less::operator()(unsigned int index) const
{
  return data.regulator(index).timestep() >= K;
}
//-----------------------------------------------------------------------------
