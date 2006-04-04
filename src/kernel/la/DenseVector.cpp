// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-04
// Last changed: 

#include <dolfin/DenseVector.h>
#include <dolfin/dolfin_log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseVector::DenseVector() : boost::numeric::ublas::vector<double>(),
    Variable("x", "a dense vector")

{
  //Do nothing
}
//-----------------------------------------------------------------------------
DenseVector::DenseVector(uint i) : boost::numeric::ublas::vector<double>(i),
    Variable("x", "a dense vector")
{
  //Do nothing
}
//-----------------------------------------------------------------------------
DenseVector::DenseVector(DenseVector& x) : boost::numeric::ublas::vector<double>(x),
    Variable("x", "a dense vector")
{
  //Do nothing
}
//-----------------------------------------------------------------------------
DenseVector::~DenseVector()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
