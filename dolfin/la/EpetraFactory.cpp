// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
// Last changed: 2008-09-28

#ifdef HAS_TRILINOS

#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>

#include "dolfin/main/MPI.h"
#include "EpetraSparsityPattern.h"
#include "SparsityPattern.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraFactory.h"

using namespace dolfin;

// Singleton instance
EpetraFactory EpetraFactory::factory;

//-----------------------------------------------------------------------------
EpetraFactory::EpetraFactory()
{
  serial_comm = new Epetra_SerialComm();

  // Why does this not work with dolfin::MPICommunicator?
  mpi_comm = new Epetra_MpiComm(MPI_COMM_WORLD);
}
//-----------------------------------------------------------------------------
EpetraFactory::~EpetraFactory()
{
  delete serial_comm;
  delete mpi_comm;
}
//-----------------------------------------------------------------------------
EpetraMatrix* EpetraFactory::create_matrix() const
{
  return new EpetraMatrix();
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraFactory::create_vector() const
{
  return new EpetraVector("global");
}
//-----------------------------------------------------------------------------
EpetraVector* EpetraFactory::create_local_vector() const
{
  return new EpetraVector("local");
}
//-----------------------------------------------------------------------------
SparsityPattern* EpetraFactory::create_pattern() const
{
  return new SparsityPattern;
}
//-----------------------------------------------------------------------------
EpetraLUSolver* EpetraFactory::create_lu_solver() const
{
  return new EpetraLUSolver();
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver* EpetraFactory::create_krylov_solver(std::string method,
                                                          std::string pc) const
{
  return new EpetraKrylovSolver(method, pc);
}
//-----------------------------------------------------------------------------
Epetra_SerialComm& EpetraFactory::get_serial_comm() const
{
  return *serial_comm;
}
//-----------------------------------------------------------------------------
Epetra_MpiComm& EpetraFactory::get_mpi_comm() const
{
  return *mpi_comm;
}
//-----------------------------------------------------------------------------
#endif
