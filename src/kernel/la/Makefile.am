SUBDIRS                = dolfin
include_HEADERS        = dolfin/dolfin_la.h \
                         dolfin/DenseMatrix.h \
                         dolfin/DirectSolver.h \
                         dolfin/GaussSeidel.h \
                         dolfin/GenericMatrix.h \
                         dolfin/KrylovSolver.h \
                         dolfin/LinearSolver.h \
                         dolfin/Matrix.h \
                         dolfin/Preconditioner.h \
                         dolfin/SISolver.h \
                         dolfin/SparseMatrix.h \
                         dolfin/Vector.h
INCLUDES               = @MODULE_INCLUDES@
lib_LIBRARIES          = libdolfin-la.a
libdolfin_la_a_SOURCES = DenseMatrix.cpp \
                         DirectSolver.cpp \
                         GaussSeidel.cpp \
                         GenericMatrix.cpp \
	                 KrylovSolver.cpp \
                         LinearSolver.cpp \
                         Matrix.cpp \
	                 SISolver.cpp \
                         SparseMatrix.cpp \
                         Vector.cpp
