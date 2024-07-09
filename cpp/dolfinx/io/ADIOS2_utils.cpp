// Copyright (C) 2021-2023 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2_utils.h"
#include <adios2.h>
#include <mpi.h>

using namespace dolfinx::io;

//-----------------------------------------------------------------------------
ADIOS2Engine::ADIOS2Engine(MPI_Comm comm, const std::filesystem::path& filename,
                           std::string tag, std::string engine,
                           const adios2::Mode mode)

    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag)))
{
  _io->SetEngine(engine);
  _engine = std::make_unique<adios2::Engine>(_io->Open(filename, mode));
}
//-----------------------------------------------------------------------------
ADIOS2Engine::~ADIOS2Engine() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2Engine::close()
{
  assert(_engine);
  if (*_engine)
    _engine->Close();
}
// //-----------------------------------------------------------------------------
// std::unique_ptr<adios2::IO> ADIOS2Engine::io()
// {
//   assert(_io);
//   if (*_io)
//     return _io;
// }
// //-----------------------------------------------------------------------------
// std::unique_ptr<adios2::Engine> ADIOS2Engine::engine()
// {
//   assert(_engine);
//   if (*_engine)
//     return _engine;
// }

#endif
