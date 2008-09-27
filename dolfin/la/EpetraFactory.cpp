// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21

#ifdef HAS_TRILINOS

#include "EpetraSparsityPattern.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraFactory.h"

#include <Epetra_SerialComm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraFactory:: EpetraFactory() {
  comm = new Epetra_SerialComm(); 
}
//-----------------------------------------------------------------------------
EpetraFactory:: ~EpetraFactory() {
  delete comm;
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraFactory::create_matrix() const 
{ 
  return new EpetraMatrix();
}
//-----------------------------------------------------------------------------
EpetraSparsityPattern* EpetraFactory::create_pattern() const 
{
  return new EpetraSparsityPattern(); 
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraFactory::create_vector() const 
{ 
  return new EpetraVector(); 
}
//-----------------------------------------------------------------------------
Epetra_SerialComm& EpetraFactory::getSerialComm()
{
  return *comm;
};
//-----------------------------------------------------------------------------

// Singleton instance
EpetraFactory EpetraFactory::factory;


#endif
