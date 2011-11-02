// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011
//
// First added:  2008-04-21
// Last changed: 2011-11-02

#ifdef HAS_TRILINOS

#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>

#include "dolfin/common/MPI.h"
#include "dolfin/common/SubSystemsManager.h"
#include "EpetraSparsityPattern.h"
#include "SparsityPattern.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraFactory.h"

#include "EpetraKrylovSolver.h"

using namespace dolfin;

// Singleton instance
EpetraFactory EpetraFactory::factory;

//-----------------------------------------------------------------------------
EpetraFactory::EpetraFactory()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EpetraFactory::~EpetraFactory()
{
  // Do nothing
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
EpetraLUSolver* EpetraFactory::create_lu_solver(std::string method) const
{
  return new EpetraLUSolver(method);
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver* EpetraFactory::create_krylov_solver(std::string method,
                                              std::string preconditioner) const
{
  return new EpetraKrylovSolver(method, preconditioner);
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraFactory::lu_solver_methods() const
{
  return EpetraLUSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraFactory::krylov_solver_methods() const
{
  return EpetraKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraFactory::krylov_solver_preconditioners() const
{
  return EpetraKrylovSolver::preconditioners();
}
//-----------------------------------------------------------------------------
Epetra_SerialComm& EpetraFactory::get_serial_comm()
{
  if (!serial_comm)
  {
    serial_comm.reset(new Epetra_SerialComm());
    assert(serial_comm);
  }
  return *serial_comm;
}
//-----------------------------------------------------------------------------
Epetra_MpiComm& EpetraFactory::get_mpi_comm()
{
  if (!mpi_comm)
  {
    mpi_comm.reset(new Epetra_MpiComm(MPI_COMM_WORLD));
    assert(mpi_comm);
  }
  return *mpi_comm;
}
//-----------------------------------------------------------------------------
#endif
