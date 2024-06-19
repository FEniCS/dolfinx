// Copyright (C) year authors
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <mpi.h>
#include <adios2.h>
#include <dolfinx/mesh/Mesh.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

void write(MPI_Comm comm, const std::filesystem::path& filename,
           std::string tag, std::shared_ptr<mesh::Mesh<float>> mesh)
           {
    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO(tag);
    adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

    const std::string mesh_name = mesh->name;
    const std::int16_t mesh_dim = mesh->geometry().dim();
    adios2::Variable<std::string> name = io.DefineVariable<std::string>("name");
    adios2::Variable<std::int16_t> dim = io.DefineVariable<std::int16_t>("dim");
    writer.BeginStep();
    writer.Put(name, mesh_name);
    writer.Put(dim, mesh_dim);
    writer.EndStep();
    writer.Close();

}


}