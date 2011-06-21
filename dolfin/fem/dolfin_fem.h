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
#include <dolfin/fem/PeriodicBC.h>
#include <dolfin/fem/EqualityBC.h>
#include <dolfin/fem/PointSource.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/fem/VariationalProblem.h>

#include <dolfin/fem/OpenMpAssembler.h>

#endif
