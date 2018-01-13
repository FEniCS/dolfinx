// Copyright (C) 2017 Tormod Landet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"

using namespace dolfin_wrappers;

//-----------------------------------------------------------------------------
MPICommWrapper::MPICommWrapper() : _comm(MPI_COMM_NULL)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPICommWrapper::MPICommWrapper(MPI_Comm comm) : _comm(comm)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MPICommWrapper& MPICommWrapper::operator=(const MPI_Comm comm)
{
  this->_comm = comm;
  return *this;
}
//-----------------------------------------------------------------------------
MPI_Comm MPICommWrapper::get() const { return _comm; }
//-----------------------------------------------------------------------------
