#pragma once

namespace dolfinx
{
/*! \namespace dolfinx::la
    \brief Linear algebra interface

    Interface to linear algebra data structures and solvers
*/
}

// DOLFINx la interface

#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScOperator.h>
#include <dolfinx/la/PETScOptions.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SLEPcEigenSolver.h>
#include <dolfinx/la/SparsityPattern.h>
