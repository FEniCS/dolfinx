// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-06
// Last changed: 2007-12-07

#include "uBlasFactory.h"

using namespace dolfin;

uBlasMatrix<ublas_sparse_matrix>* uBlasFactory::createMatrix() const 
{
  return new uBlasMatrix<ublas_sparse_matrix>(); 
}

SparsityPattern* uBlasFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

uBlasVector* uBlasFactory::createVector() const 
{
  return new uBlasVector(); 
}

uBlasFactory uBlasFactory::ublasfactory;
