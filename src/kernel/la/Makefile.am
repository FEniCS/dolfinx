SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_la.h \
                          dolfin/DenseMatrix.h \
                          dolfin/DenseVector.h \
                          dolfin/EigenvalueSolver.h \
                          dolfin/KrylovSolver.h \
                          dolfin/GenericMatrix.h \
                          dolfin/GenericVector.h \
                          dolfin/GMRES.h \
                          dolfin/LinearSolver.h \
                          dolfin/LU.h \
                          dolfin/Matrix.h \
                          dolfin/PETScKrylovSolver.h \
                          dolfin/PETScLU.h \
                          dolfin/PETScManager.h \
                          dolfin/PETScSparseMatrix.h \
                          dolfin/PETScVector.h \
                          dolfin/Preconditioner.h \
                          dolfin/SparseMatrix.h \
                          dolfin/ublas.h \
                          dolfin/uBlasKrylovSolver.h \
                          dolfin/uBlasLUSolver.h \
                          dolfin/uBlasPreconditioner.h \
                          dolfin/uBlasSparseMatrix.h \
                          dolfin/Vector.h \
                          dolfin/VirtualMatrix.h 
INCLUDES                = @KERNEL_CFLAGS@
lib_LTLIBRARIES         = libdolfin-la.la
libdolfin_la_la_SOURCES = DenseMatrix.cpp \
                          DenseVector.cpp \
                          EigenvalueSolver.cpp \
                          GMRES.cpp \
                          LinearSolver.cpp \
                          PETScKrylovSolver.cpp \
                          PETScLU.cpp \
                          PETScManager.cpp \
                          PETScSparseMatrix.cpp \
                          PETScVector.cpp \
                          Preconditioner.cpp \
                          uBlasKrylovSolver.cpp \
                          uBlasLUSolver.cpp \
                          uBlasPreconditioner.cpp \
                          uBlasSparseMatrix.cpp \
                          VirtualMatrix.cpp 
libdolfin_la_la_LDFLAGS = -static
