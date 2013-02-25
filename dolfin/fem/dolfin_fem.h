#ifndef __DOLFIN_FEM_H
#define __DOLFIN_FEM_H

// DOLFIN fem interface

#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Equation.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/BasisFunction.h>
#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/PointSource.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/LocalSolver.h>
#include <dolfin/fem/solve.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/AssemblerBase.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalSolver.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/fem/NonlinearVariationalSolver.h>
#include <dolfin/fem/ButcherScheme.h>
#include <dolfin/fem/RKSolver.h>

// Move up when ready or merge with Assembler.h
#include <dolfin/fem/OpenMpAssembler.h>

// Remove when no longer needed, here to give deprecation warning
#include <dolfin/fem/VariationalProblem.h>

#endif
