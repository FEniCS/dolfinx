// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DOLFIN_ODE_H
#define __DOLFIN_ODE_H

// Multi-adaptivity

#include <dolfin/cGqElement.h>
#include <dolfin/cGqMethod.h>
#include <dolfin/cGqMethods.h>
#include <dolfin/dGqElement.h>
#include <dolfin/dGqMethod.h>
#include <dolfin/dGqMethods.h>
#include <dolfin/AdaptiveIterationLevel1.h>
#include <dolfin/AdaptiveIterationLevel2.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/ComplexODE.h>
#include <dolfin/Component.h>
#include <dolfin/Dual.h>
#include <dolfin/Element.h>
#include <dolfin/ElementData.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/Homotopy.h>
#include <dolfin/HomotopyJacobian.h>
#include <dolfin/HomotopyODE.h>
#include <dolfin/Iteration.h>
#include <dolfin/JacobianMatrix.h>
#include <dolfin/Method.h>
#include <dolfin/NonStiffIteration.h>
#include <dolfin/ODE.h>
#include <dolfin/ODESolver.h>
#include <dolfin/ParticleSystem.h>
#include <dolfin/NewParticleSystem.h>
#include <dolfin/RHS.h>
#include <dolfin/RecursiveTimeSlab.h>
#include <dolfin/ReducedModel.h>
#include <dolfin/Regulator.h>
#include <dolfin/Sample.h>
#include <dolfin/NewSample.h>
#include <dolfin/Solution.h>
#include <dolfin/Sparsity.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/TimeStepper.h>

#endif
