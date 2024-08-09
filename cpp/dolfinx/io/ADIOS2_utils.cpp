// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2_utils.h"
#include <adios2.h>
#include <mpi.h>

namespace
{
std::map<std::string, adios2::Mode> string_to_mode{
    {"write", adios2::Mode::Write},
    {"read", adios2::Mode::Read},
};
} // namespace

using namespace dolfinx::io;

//-----------------------------------------------------------------------------
ADIOS2Wrapper::ADIOS2Wrapper(MPI_Comm comm, std::string filename,
                             std::string tag, std::string engine_type,
                             std::string mode)

    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag)))
{
  _io->SetEngine(engine_type);
  _engine = std::make_unique<adios2::Engine>(
      _io->Open(filename, string_to_mode[mode]));
}
//-----------------------------------------------------------------------------
ADIOS2Wrapper::~ADIOS2Wrapper() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2Wrapper::close()
{
  assert(_engine);
  if (*_engine)
    _engine->Close();
}

#endif