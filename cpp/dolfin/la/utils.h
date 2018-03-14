// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

namespace dolfin
{
namespace la
{
/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);
}
}
