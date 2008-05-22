#ifndef __DOLFIN_LA_H
#define __DOLFIN_LA_H

// DOLFIN la interface

// Note that the order is important!

#include <dolfin/la/VectorNormType.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/la/uBlasMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/EpetraMatrix.h>
#include <dolfin/la/uBlasVector.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/EpetraVector.h>
#include <dolfin/la/AssemblyMatrix.h>
#include <dolfin/la/GenericSparsityPattern.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/la/uBlasFactory.h>
#include <dolfin/la/PETScFactory.h>
#include <dolfin/la/EpetraFactory.h>
#include <dolfin/la/AssemblyFactory.h>
#include <dolfin/la/SolverType.h>
#include <dolfin/la/PreconditionerType.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/SLEPcEigenvalueSolver.h>
#include <dolfin/la/uBlasDenseMatrix.h>
#include <dolfin/la/uBlasKrylovSolver.h>
#include <dolfin/la/uBlasLUSolver.h>
#include <dolfin/la/uBlasPreconditioner.h>
#include <dolfin/la/uBlasILUPreconditioner.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/SingularSolver.h>
#include <dolfin/la/solve.h>

#endif
