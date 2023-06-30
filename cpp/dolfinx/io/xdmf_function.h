// Copyright (C) 2012-2023 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <complex>
#include <concepts>
#include <dolfinx/common/types.h>
#include <hdf5.h>
#include <mpi.h>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{
namespace fem
{
template <dolfinx::scalar T, std::floating_point U>
class Function;
}

/// Low-level methods for reading/writing XDMF files
namespace io::xdmf_function
{

/// Write a fem::Function to XDMF
template <dolfinx::scalar T, std::floating_point U>
void add_function(MPI_Comm comm, const fem::Function<T, U>& u, double t,
                  pugi::xml_node& xml_node, const hid_t h5_id);
} // namespace io::xdmf_function
} // namespace dolfinx
