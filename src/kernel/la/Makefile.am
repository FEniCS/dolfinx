SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_la.h \
                          dolfin/solve.h \
                          dolfin/AssemblyMatrix.h \
                          dolfin/DenseMatrix.h \
                          dolfin/DenseVector.h \
                          dolfin/KrylovMethod.h \
                          dolfin/KrylovSolver.h \
                          dolfin/GenericMatrix.h \
                          dolfin/GenericVector.h \
                          dolfin/GenericTensor.h \
                          dolfin/GMRES.h \
                          dolfin/LinearSolver.h \
                          dolfin/LU.h \
                          dolfin/LUSolver.h \
                          dolfin/Matrix.h \
                          dolfin/default_la_types.h \
                          dolfin/NewKrylovSolver.h \
                          dolfin/NewLinearSolver.h \
                          dolfin/NewLUSolver.h \
                          dolfin/NewMatrix.h \
                          dolfin/NewVector.h \
                          dolfin/SLEPcEigenvalueSolver.h \
                          dolfin/PETScKrylovMatrix.h \
                          dolfin/PETScKrylovSolver.h \
                          dolfin/PETScLinearSolver.h \
                          dolfin/PETScLUSolver.h \
                          dolfin/PETScManager.h \
                          dolfin/PETScMatrix.h \
                          dolfin/PETScPreconditioner.h \
                          dolfin/PETScVector.h \
                          dolfin/Preconditioner.h \
                          dolfin/SparsityPattern.h \
                          dolfin/ublas.h \
                          dolfin/uBlasDenseMatrix.h \
                          dolfin/uBlasKrylovMatrix.h \
                          dolfin/uBlasKrylovSolver.h \
                          dolfin/uBlasLinearSolver.h \
                          dolfin/uBlasLUSolver.h \
                          dolfin/uBlasMatrix.h \
                          dolfin/uBlasDummyPreconditioner.h \
                          dolfin/uBlasILUPreconditioner.h \
                          dolfin/uBlasPreconditioner.h \
                          dolfin/uBlasSparseMatrix.h \
                          dolfin/uBlasVector.h \
	                  dolfin/Scalar.h \
                          dolfin/Vector.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES         = libdolfin-la.la
libdolfin_la_la_SOURCES = solve.cpp \
                          GMRES.cpp \
                          LU.cpp \
                          SLEPcEigenvalueSolver.cpp \
                          PETScKrylovMatrix.cpp \
                          PETScKrylovSolver.cpp \
                          PETScLUSolver.cpp \
                          PETScManager.cpp \
                          PETScMatrix.cpp \
                          PETScPreconditioner.cpp \
                          PETScVector.cpp \
                          SparsityPattern.cpp \
                          uBlasKrylovMatrix.cpp \
                          uBlasKrylovSolver.cpp \
                          uBlasLUSolver.cpp \
                          uBlasDummyPreconditioner.cpp \
                          uBlasILUPreconditioner.cpp \
                          uBlasVector.cpp
libdolfin_la_la_LDFLAGS = -static
