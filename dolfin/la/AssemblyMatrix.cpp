// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2008-05-22
// Last changed: 2008-05-22


#include <vector>
#include <map>
#include <sstream>
#include <iomanip>

#include <dolfin/common/Array.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/common/Variable.h>
#include "LinearAlgebraFactory.h"
#include "SparsityPattern.h"
#include "AssemblyFactory.h"
#include "uBlasVector.h"
#include "GenericMatrix.h"

using namespace dolfin;


//-----------------------------------------------------------------------------
AssemblyMatrix::AssemblyMatrix(): 
  dims(0)
{ 
  dims = new uint[2]; 
}
//-----------------------------------------------------------------------------
AssemblyMatrix::AssemblyMatrix(uint M, uint N) : 
  dims(0)
{ 
  dims = new uint[2]; 
  init(M, N);
}
//-----------------------------------------------------------------------------
AssemblyMatrix::AssemblyMatrix(const AssemblyMatrix& A):
  dims(0)
{ 
  error("AssemblyMatrix: Not implemented.");
}
//-----------------------------------------------------------------------------
AssemblyMatrix::~AssemblyMatrix()
{ 
  delete [] dims;
  A.clear();
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::init(uint M, uint N)
{
  dims[0] = M;
  dims[1] = N;
  A.clear();
  // Set number of rows
  A.clear();
  A.resize(M);
  /*
  // Initialize with zeros
  for (uint i = 0; i < M; i++)
  for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
  it->second = 0.0;
  */
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
      uint M = sparsity_pattern.size(0);
      uint N = sparsity_pattern.size(1);
      init(M, N);
      /// The comment directly below also makes the following 
      /// two lines unnecessary:
      /*
      uint* nzrow = new uint[M];
      sparsity_pattern.numNonZeroPerRow(nzrow);
      */
      /// Map does not support reserve. Also, allocating space for a std::map
      /// is not needed as it does not need to copy an element because others 
      /// are inserted.
      /*
      for (uint i = 0; i < M; ++i)
        A[i].reserve(nzrow[i]);
      delete [] nzrow;
      */
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::init(uint rank, const uint* dims, bool reset)
{
  // Check that the rank is 2
  if ( rank != 2 )
    error("Illegal tensor rank (%d) for matrix. Rank must be 2.", rank);
  
  // Initialize matrix
  init(dims[0], dims[1]);
  
  // Save dimensions
  this->dims[0] = dims[0];
  this->dims[1] = dims[1];
}
//-----------------------------------------------------------------------------
AssemblyMatrix* AssemblyMatrix::copy() const
{ 
  AssemblyMatrix* mcopy = AssemblyFactory::instance().createMatrix();
  error("copy: Not implemented.");
  return mcopy;
}
//-----------------------------------------------------------------------------
dolfin::uint AssemblyMatrix::size(uint dim) const
{ 
  return dims[dim]; 
}

/*
void AssemblyMatrix::get(real* block, 
                         const uint* num_rows, 
                         const uint * const * rows) const
{ 
  get(block, num_rows[0], rows[0], num_rows[1], rows[1]); 
}
*/

//-----------------------------------------------------------------------------
void AssemblyMatrix::get(real* block, 
                         uint m, const uint* rows, 
                         uint n, const uint* cols) const
{ 
  error("get: Not implemented"); 
}

/*
void AssemblyMatrix::set(const real* block, 
                         const uint* num_rows, 
                         const uint * const * rows)
{ 
  set(block, num_rows[0], rows[0], num_rows[1], rows[1]); 
}
*/

//-----------------------------------------------------------------------------
void AssemblyMatrix::set(const real* block, 
                         uint m, const uint* rows, 
                         uint n, const uint* cols)
{ 
  error("set: Not implemented"); 
}

/*
void AssemblyMatrix::add(const real* block, 
                         const uint* num_rows, 
                         const uint * const * rows)
{ 
  add(block, num_rows[0], rows[0], num_rows[1], rows[1]); 
}
*/

//-----------------------------------------------------------------------------
void AssemblyMatrix::add(const real* block, 
                         uint m, const uint* rows, 
                         uint n, const uint* cols)
{
  uint pos = 0;
  for (uint i = 0; i < m; i++)
    {
      std::map<uint, real>& row = A[rows[i]];
      for (uint j = 0; j < n; j++)
        {
          const uint col = cols[j];
          const std::map<uint, real>::iterator it = row.find(col);
          if ( it == row.end() )
            row.insert(it, std::map<uint, real>::value_type(col, block[pos++]));
          else
            it->second += block[pos++];
        }
    }
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::zero()
{
  for (uint i = 0; i < A.size(); i++)
    for (std::map<uint, real>::iterator it = A[i].begin(); it != A[i].end(); it++)
      it->second = 0.0;
} 
//-----------------------------------------------------------------------------
void AssemblyMatrix::apply(FinalizeType finaltype)
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::disp(uint precision) const
{
  for (uint i = 0; i < dims[0]; i++)
    {
      std::stringstream line;
      line << std::setiosflags(std::ios::scientific);
      line << std::setprecision(precision);
      
      line << "|";
      for (std::map<uint, real>::const_iterator it = A[i].begin(); it != A[i].end(); it++)
        line << " (" << i << ", " << it->first << ", " << it->second << ")";
      line << " |";
      
      dolfin::cout << line.str().c_str() << dolfin::endl;
    }
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::ident(uint m, const uint* rows)
{ 
  error("ident: Not implemented."); 
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::zero(uint m, const uint* rows)
{ 
  error("zero: Not implemented."); 
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::mult(const GenericVector& x, GenericVector& y, bool transposed) const
{ 
  error("mult: Not implemented."); 
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::getrow(uint row, Array<uint>& columns, Array<real>& values) const
{ 
  //dolfin_assert(A);
  //message("getrow: Not implemented.");
  //error("getrow: Not implemented."); 
  columns.clear();
  values.clear();
  //message("%d", row);
  const std::map<uint, real>& rowid = A[row];
  for (std::map<uint, real>::const_iterator it = rowid.begin(); it != rowid.end(); it++)
    {
      columns.push_back(it->first);
      values.push_back(it->second);
    }
  //  int k = A[row].size();
  // message("%d", row);
}
//-----------------------------------------------------------------------------
void AssemblyMatrix::setrow(uint row, const Array<uint>& columns, const Array<real>& values)
{ 
  error("setrow: Not implemented."); 
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& AssemblyMatrix::factory() const
{ 
  return AssemblyFactory::instance(); 
}
//-----------------------------------------------------------------------------
const AssemblyMatrix& AssemblyMatrix::operator*= (real a)
{ 
  error("operator*=: Not implemented.");
  return *this;
}
//-----------------------------------------------------------------------------
const AssemblyMatrix& AssemblyMatrix::operator/= (real a)
{
  error("operator/=: Not implemented.");
  return *this;
}



