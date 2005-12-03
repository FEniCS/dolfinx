SUBDIRS                = dolfin
nobase_include_HEADERS = dolfin/dolfin_la.h \
                         dolfin/EigenvalueSolver.h \
                         dolfin/KrylovSolver.h \
                         dolfin/GMRES.h \
                         dolfin/LinearSolver.h \
                         dolfin/LU.h \
                         dolfin/Matrix.h \
                         dolfin/PETScManager.h \
                         dolfin/Preconditioner.h \
                         dolfin/Vector.h \
                         dolfin/VirtualMatrix.h 
INCLUDES               = @KERNEL_CFLAGS@
lib_LIBRARIES          = libdolfin-la.a
libdolfin_la_a_SOURCES = EigenvalueSolver.cpp \
                         KrylovSolver.cpp \
                         GMRES.cpp \
                         LinearSolver.cpp \
                         LU.cpp \
                         Matrix.cpp \
                         PETScManager.cpp \
                         Preconditioner.cpp \
                         Vector.cpp \
                         VirtualMatrix.cpp 
