// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-13
// Last changed:

#include <dolfin/dolfin_log.h>
#include <dolfin/SparsityPattern.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern()
{
  sparsity_pattern.clear();
  dim[0] = 0;
  dim[1] = 0;
}
//-----------------------------------------------------------------------------
SparsityPattern::~SparsityPattern()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(const uint M, const uint N)
{
  dim[0] = M;
  dim[1] = N;
  sparsity_pattern.clear();
  sparsity_pattern.resize( M );
}
//-----------------------------------------------------------------------------
void SparsityPattern::numNonZeroPerRow(uint nzrow[]) const
{
  if ( sparsity_pattern.size() == 0 )
    dolfin_error("Sparsity pattern has not been computed.");

  // Compute number of nonzeros per row
  std::vector< std::set<int> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nzrow[set-sparsity_pattern.begin()] = set->size();
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::numNonZero() const
{
  if ( sparsity_pattern.size() == 0 )
    dolfin_error("Sparsity pattern has not been computed.");

  // Compute total number of nonzeros per row
  uint nz = 0;
  std::vector< std::set<int> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nz += set->size();
  return nz;
}
//-----------------------------------------------------------------------------
void SparsityPattern::disp() const
{ 
  std::vector< std::set<int> >::const_iterator set;
  std::set<int>::const_iterator element;
  
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
  {
    cout << "Row " << endl;
    for(element = set->begin(); element != set->end(); ++element)
      cout << *element << " ";
    cout << endl;
  }  
}
//-----------------------------------------------------------------------------
