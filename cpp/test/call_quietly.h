// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <dolfinx/common/log.h>
#include <functional>
#include <petscsys.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>

namespace dolfinx_test
{
/// @brief Call `f`, which is expected to fail or warn, without PETSc's
/// error traceback or DOLFINx logging.
///
/// The PETSc error handler and the DOLFINx log level are restored
/// before an exception from `f` propagates.
/// @param[in] f Callable to invoke.
template <std::invocable F>
void call_quietly(F&& f)
{
  if (PetscPushErrorHandler(PetscReturnErrorHandler, nullptr) != 0)
    throw std::runtime_error("PetscPushErrorHandler failed");
  spdlog::logger& log = dolfinx::logger();
  const spdlog::level::level_enum level = log.level();
  log.set_level(spdlog::level::off);
  try
  {
    std::invoke(std::forward<F>(f));
  }
  catch (...)
  {
    log.set_level(level);
    static_cast<void>(PetscPopErrorHandler());
    throw;
  }
  log.set_level(level);
  if (PetscPopErrorHandler() != 0)
    throw std::runtime_error("PetscPopErrorHandler failed");
}
} // namespace dolfinx_test
