#pragma once

namespace dolfinx
{
/*! \namespace dolfinx::fem
    \brief Finite element method functionality

    Classes and algorithms for finite element method operations, e.g. assembly.
*/
}

// DOLFINX fem interface

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DiscreteOperators.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/PETScDMCollection.h>
#include <dolfinx/fem/SparsityPatternBuilder.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
