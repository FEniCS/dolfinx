// Copyright (C) 2004-2026 Johan Hoffman, Johan Jansson, Anders Logg,
// Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "petsc.h"
#include "petscsystypes.h"
#include <cassert>
#include <format>
#include <stdexcept>
#include <utility>

using namespace dolfinx;

//-----------------------------------------------------------------------------
void common::petsc::error(PetscErrorCode error_code,
                          std::string_view petsc_function,
                          std::source_location loc)
{
  // Only called from check() with a nonzero error code
  assert(error_code);

  // PETSc's specific message may belong to an earlier failure.
  const char* desc = nullptr;
  if (PetscErrorMessage(error_code, &desc, nullptr) != PETSC_SUCCESS or !desc)
    desc = "(no error description available)";

  throw std::runtime_error(std::format(
      "PETSc error in '{}' at {}:{}: call to '{}' failed with code {}, {}",
      loc.function_name(), loc.file_name(), loc.line(), petsc_function,
      static_cast<int>(error_code), desc));
}
//-----------------------------------------------------------------------------
void common::petsc::set_option(std::string option)
{
  common::petsc::set_option<std::string>(std::move(option), "");
}
//-----------------------------------------------------------------------------
void common::petsc::clear_option(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  check(PetscOptionsClearValue(nullptr, option.c_str()),
        "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void common::petsc::clear_options()
{
  check(PetscOptionsClear(nullptr), "PetscOptionsClear");
}
//-----------------------------------------------------------------------------

#endif
