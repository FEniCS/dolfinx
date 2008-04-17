// Copyright (C) 2008 Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-24
// Last changed: 2008-01-24

#ifdef HAS_TRILINOS

#include "SparsityPattern.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraFactory.h"

using namespace dolfin;

EpetraMatrix* EpetraFactory::createMatrix() const 
{ 
  return new EpetraMatrix();
}

SparsityPattern* EpetraFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

EpetraVector* EpetraFactory::createVector() const 
{ 
  return new EpetraVector(); 
}

EpetraFactory EpetraFactory::epetrafactory;

#endif
