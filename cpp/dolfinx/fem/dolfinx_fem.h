#pragma once

/// @brief Finite element method functionality
///
/// Classes and algorithms for finite element method spaces and
/// operations.
namespace dolfinx::fem
{
}

// DOLFINx fem interface

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/expression_evaluate.h>
#include <dolfinx/fem/expression_factory.h>
#include <dolfinx/fem/form_factory.h>
#include <dolfinx/fem/functionspace_factory.h>
#include <dolfinx/fem/integration_domains.h>
#include <dolfinx/fem/interpolate_geometry.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/sparsitypattern.h>
