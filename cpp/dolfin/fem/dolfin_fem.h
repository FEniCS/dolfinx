#pragma once

namespace dolfin
{
/*! \namespace dolfin::fem
    \brief Finite element method functionality

    Classes and algorithms for finite element method operations, e.g. assembly.
*/
}

// DOLFIN fem interface

#include <dolfin/fem/assembler.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DiscreteOperators.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/PETScDMCollection.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/fem/utils.h>
