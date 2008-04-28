// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21

#ifdef HAS_TRILINOS

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "EpetraFactory.h"
#include "GenericSparsityPattern.h"
#include "EpetraSparsityPattern.h"


#include <Epetra_SerialComm.h>
#include <Epetra_FECrsGraph.h>

using namespace dolfin;
using dolfin::uint;

//-----------------------------------------------------------------------------
EpetraSparsityPattern::EpetraSparsityPattern() : epetra_graph(0), rank(0), dims(0) { 
}
//-----------------------------------------------------------------------------
EpetraSparsityPattern::~EpetraSparsityPattern() { 
  delete epetra_graph; 
  delete dims; 
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::init(uint rank_, const uint* dims_){
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

    EpetraFactory& f = dynamic_cast<EpetraFactory&>(factory());
    Epetra_SerialComm Comm = f.getSerialComm();

    Epetra_Map row_map(dims[0], 0, Comm);
    epetra_graph = new Epetra_FECrsGraph(Copy, row_map, 0);   
  }
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::pinit(uint rank, const uint* dims){
  error("EpetraSparsityPattern::pinit not implemented yet."); 
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::insert(const uint* num_rows, const uint * const * rows){
  
  if (rank == 2) {
    epetra_graph->InsertGlobalIndices(
        num_rows[0], reinterpret_cast<const int*>(rows[0]), 
        num_rows[1], reinterpret_cast<const int*>(rows[1]));
  }
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::pinsert(const uint* num_rows, const uint * const * rows){
  error("EpetraSparsityPattern::pinsert not implemented yet."); 
}
//-----------------------------------------------------------------------------
uint EpetraSparsityPattern::size(uint n) const { 
  if (rank == 1) {
    return dims[0]; 
  }
  if (rank == 2 ) {
    dolfin_assert(epetra_graph); 
    if ( n==0) {
      return epetra_graph->NumGlobalRows(); 
    } else {
      return epetra_graph->NumGlobalCols(); 
    }
  }
  return 0; 
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::numNonZeroPerRow(uint nzrow[]) const { 
  error("EpetraSparsityPattern::numNonZeroPerRow not implemented yet"); 
}
//-----------------------------------------------------------------------------
uint EpetraSparsityPattern::numNonZero() const {
  dolfin_assert(epetra_graph); 
  return epetra_graph->NumGlobalNonzeros(); 
}
//-----------------------------------------------------------------------------
void EpetraSparsityPattern::apply() {
  dolfin_assert(epetra_graph); 

  // Could employ eg. OptimizeStorage. Not sure if this is wanted, 
  // the graph would then depend on the equations, not only the method. 

  EpetraFactory& f = dynamic_cast<EpetraFactory&>(factory());
  Epetra_SerialComm Comm = f.getSerialComm();

  Epetra_Map row_map(dims[0], 0, Comm);
  Epetra_Map col_map(dims[1], 0, Comm);

  epetra_graph->FillComplete (col_map, row_map);

}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& EpetraSparsityPattern::factory() const
{
  return EpetraFactory::instance();
}
//-----------------------------------------------------------------------------
Epetra_FECrsGraph& EpetraSparsityPattern:: pattern() const 
{
  return *epetra_graph; 
}
//-----------------------------------------------------------------------------
#endif




