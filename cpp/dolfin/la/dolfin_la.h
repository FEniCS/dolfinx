#pragma once

namespace dolfin
{
/*! \namespace dolfin::la
    \brief Linear algebra interface

    Interface to linear algebra data structures and solvers
*/
}

// DOLFIN la interface

#include <dolfin/la/PETScBaseMatrix.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/VectorSpaceBasis.h>
