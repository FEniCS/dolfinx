// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <hdf5.h>
#include <mpi.h>
#include <petscsys.h>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{
namespace function
{
template <typename T>
class Function;
}

namespace io
{
/// Low-level methods for reading/writing XDMF files
namespace xdmf_function
{

/// TODO
void add_function(MPI_Comm comm, const function::Function<PetscScalar>& u,
                  const double t, pugi::xml_node& xml_node, const hid_t h5_id);

} // namespace xdmf_function
} // namespace io
} // namespace dolfinx
