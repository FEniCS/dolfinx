SUBDIRS                = dolfin
include_HEADERS        = dolfin/dolfin_la.h \
                         dolfin/DenseMatrix.h \
                         dolfin/DirectSolver.h \
                         dolfin/GenericMatrix.h \
                         dolfin/KrylovSolver.h \
                         dolfin/Matrix.h \
                         dolfin/SISolver.h \
                         dolfin/SparseMatrix.h \
                         dolfin/Vector.h \
                         dolfin/VirtualMatrix.h
INCLUDES               = @MODULE_INCLUDES@
lib_LIBRARIES          = libdolfin-la.a
libdolfin_la_a_SOURCES = DenseMatrix.cpp \
                         DirectSolver.cpp \
                         Matrix.cpp \
                         Vector.cpp \
	                 SISolver.h \
	                 SISolver.cpp \
	                 KrylovSolver.cpp \
                         SparseMatrix.cpp
