// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petscis.h>
#include <string>
#include <vector>

namespace dolfin
{
namespace common
{
class IndexMap;
}
namespace la
{

/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

/// Compute IndexSets (IS) for stacked index maps
std::vector<IS> compute_index_sets(std::vector<const common::IndexMap*> maps);

/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);
} // namespace la
} // namespace dolfin
