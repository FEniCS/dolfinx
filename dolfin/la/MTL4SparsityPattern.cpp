// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06
// Last changed: 2008-07-06

#ifdef HAS_MTL4

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "MTL4Factory.h"
#include "GenericSparsityPattern.h"
#include "MTL4SparsityPattern.h"

using namespace dolfin;
using dolfin::uint;

//-----------------------------------------------------------------------------
MTL4SparsityPattern::MTL4SparsityPattern() : rank(0), dims(0) { 
}
//-----------------------------------------------------------------------------
MTL4SparsityPattern::~MTL4SparsityPattern() { 
  delete dims; 
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::init(uint rank_, const uint* dims_){
  rank = rank_;  

  //init for vector 
  if (rank == 1) {
    dims = new uint(1);
    dims[0] = dims_[0]; 
  }
  //init for matrix 
  if ( rank == 2) {
    dims = new uint(2);
    dims[0] = dims_[0]; 
    dims[1] = dims_[1]; 
  }
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::pinit(uint rank, const uint* dims){
  error("MTL4SparsityPattern::pinit not implemented yet."); 
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::insert(const uint* num_rows,const uint * const * rows)
{
  // MTL suffers when the full sparsity pattern loop executes, due to the
  // overhead in iterarion.
  error("MTL4SparsityPattern::insert was called.");
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::pinsert(const uint* num_rows,
				  const uint * const * rows)
{
  error("MTL4SparsityPattern::pinsert not implemented yet."); 
}
//-----------------------------------------------------------------------------
uint MTL4SparsityPattern::size(uint n) const 
{ 
  if (rank == 1)
    return dims[0]; 
 
  if (rank == 2 )
    return dims[n];

  return 0; 
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::numNonZeroPerRow(uint nzrow[]) const 
{ 
  error("MTL4SparsityPattern::numNonZeroPerRow not implemented yet"); 
}
//-----------------------------------------------------------------------------
uint MTL4SparsityPattern::numNonZero() const 
{
  error("MTL4SparsityPattern::numNonZero not implemented yet");
  return 0;
}
//-----------------------------------------------------------------------------
void MTL4SparsityPattern::apply() 
{

}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& MTL4SparsityPattern::factory() const
{
  return MTL4Factory::instance();
}
//-----------------------------------------------------------------------------
// MTL4_FECrsGraph& MTL4SparsityPattern:: pattern() const 
// {
//   return *epetra_graph; 
// }
//-----------------------------------------------------------------------------
#endif
