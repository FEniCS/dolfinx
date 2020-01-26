#pragma once

namespace dolfinx
{
/*! \namespace dolfinx::la
    \brief Linear algebra interface

    Interface to linear algebra data structures and solvers
*/
}

// DOLFIN la interface

#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOperator.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/VectorSpaceBasis.h>
