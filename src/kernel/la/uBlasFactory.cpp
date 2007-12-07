// Copyright (C) 2005-2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-06
// Last changed: 2007-12-07

#include <dolfin/uBlasMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasFactory.h>
#include <dolfin/SparsityPattern.h>

using namespace dolfin;

GenericMatrix* dolfin::uBlasFactory::createMatrix() const 
{
  return new uBlasMatrix<ublas_dense_matrix>(); 
}

GenericSparsityPattern* dolfin::uBlasFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

GenericVector* dolfin::uBlasFactory::createVector() const 
{
  return new uBlasVector(); 
}

dolfin::uBlasFactory dolfin::uBlasFactory::ublasfactory;
