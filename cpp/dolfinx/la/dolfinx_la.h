#pragma once

/// @brief Linear algebra interface.
///
/// Interface to linear algebra data structures and solvers.
namespace dolfinx::la
{
}

// DOLFINx la interface

#include <dolfinx/la/SparsityPattern.h>
#ifdef HAS_PETSC
#include <dolfinx/la/petsc.h>
#endif
#include <dolfinx/la/slepc.h>
#include <dolfinx/la/utils.h>
