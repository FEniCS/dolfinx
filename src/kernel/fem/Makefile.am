SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/BoundaryCondition.h \
                          dolfin/FiniteElement.h \
                          dolfin/FunctionSpace.h \
                          dolfin/Galerkin.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/MultigridSolver.h \
                          dolfin/StiffnessMatrix.h
INCLUDES                = @MODULE_INCLUDES@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = FiniteElement.cpp \
                          FunctionSpace.cpp \
                          Galerkin.cpp \
                          LoadVector.cpp \
                          MassMatrix.cpp \
                          MultigridSolver.cpp \
                          StiffnessMatrix.cpp
