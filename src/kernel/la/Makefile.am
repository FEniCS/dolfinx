SUBDIRS                = dolfin
include_HEADERS        = dolfin/Vector.h dolfin/SparseMatrix.h
INCLUDES               = @MODULE_INCLUDES@
lib_LIBRARIES          = libdolfin-la.a
libdolfin_la_a_SOURCES = DenseMatrix.cpp \
                         DenseMatrix.hh \
                         DirectSolver.cpp \
                         DirectSolver.hh \
                         SparseMatrix.cpp \
                         Vector.cpp \
	                 SISolver.hh \
	                 SISolver.cpp \
	                 KrylovSolver.hh \
	                 KrylovSolver.cpp
