SUBDIRS                = dolfin
include_HEADERS        = dolfin/Vector.h dolfin/Matrix.h dolfin/DenseMatrix.h dolfin/KrylovSolver.h dolfin/DirectSolver.h
INCLUDES               = @MODULE_INCLUDES@
lib_LIBRARIES          = libdolfin-la.a
libdolfin_la_a_SOURCES = DenseMatrix.cpp \
                         DirectSolver.cpp \
                         Matrix.cpp \
                         Vector.cpp \
	                 SISolver.h \
	                 SISolver.cpp \
	                 KrylovSolver.cpp
