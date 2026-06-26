// Copyright (C) 2025 Jørgen S. Dokken and Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPC.h"
#include "assemble_mpc.h"
#include "utils.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>

// Note - this file can be removed: for development use only
namespace dolfinx::fem
{
template MPC<double, double>::MPC(const FunctionSpace<double>&,const std::vector<std::int32_t>&,
    const std::vector<std::vector<std::pair<double, std::int64_t>>>&);

// Assemble into CSR Matrix
template void assemble_mpc(
    const MPC<double, double>& mpc, la::MatrixCSR<double>& A,
    const fem::Form<double, double>& a,
    const std::vector<
        std::reference_wrapper<const DirichletBC<double, double>>>& bcs);

template void build_sparsity_pattern_mpc(la::SparsityPattern& pattern,
                                         const Form<double, double>& form,
                                         const MPC<double, double>& mpc);

} // namespace dolfinx::fem
