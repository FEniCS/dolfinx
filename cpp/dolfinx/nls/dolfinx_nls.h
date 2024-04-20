#pragma once

/// @brief Nonlinear solvers.
///
/// Methods for solving nonlinear equations.
namespace dolfinx::nls
{
}

// DOLFINx nonlinear solver

#ifdef HAS_PETSC
#include <dolfinx/nls/NewtonSolver.h>
#endif
