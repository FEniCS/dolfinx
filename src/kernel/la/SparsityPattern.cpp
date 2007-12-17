// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-13
// Last changed:

#include <dolfin/dolfin_log.h>
#include <dolfin/SparsityPattern.h>
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern::SparsityPattern() 
{
  dim[0] = 0;
  dim[1] = 0;
  sparsity_pattern.clear();
}
//-----------------------------------------------------------------------------
SparsityPattern::~SparsityPattern()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void SparsityPattern::init(uint rank, const uint* dims)
{
  dolfin_assert(rank <= 2);
  dim[0] = dim[1] = 0;
  for (int i=0; i<rank; ++i)
    dim[i] = dims[i];
  sparsity_pattern.clear();
  sparsity_pattern.resize( dim[0] );
}
//-----------------------------------------------------------------------------
void SparsityPattern::insert(const uint* num_rows, const uint * const * rows)
{ 
  for (unsigned int i = 0; i<num_rows[0];++i)
    for (unsigned int j = 0; j<num_rows[1];++j){
      sparsity_pattern[rows[0][i]].insert(rows[1][j]);
    }
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::size(uint n) const
{
  dolfin_assert(n < 2);
  return dim[n]; 
}
//-----------------------------------------------------------------------------
void SparsityPattern::numNonZeroPerRow(uint nzrow[]) const
{
  if ( dim[1] == 0 )
    error("Non-zero entries per row can be computed for matrices only.");

  if ( sparsity_pattern.size() == 0 )
    error("Sparsity pattern has not been computed.");

  // Compute number of nonzeros per row
  std::vector< std::set<int> >::const_iterator set;
  for(set = sparsity_pattern.begin(); set != sparsity_pattern.end(); ++set)
    nzrow[set-sparsity_pattern.begin()] = set->size();
}
//-----------------------------------------------------------------------------
dolfin::uint SparsityPattern::numNonZero() const
{
  if ( dim[1] == 0 )
    error("Total non-zeros entries can be computed for matrices only.");

  if ( sparsity_pattern.size() == 0 )
    error("Sparsity pattern has not been computed.");

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
  if ( dim[1] == 0 )
    warning("Only matrix sparsity patterns can be displayed.");

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
