SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/BoundaryCondition.h \
                          dolfin/FEM.h \
                          dolfin/FiniteElement.h \
                          dolfin/FiniteElementMethod.h \
                          dolfin/FunctionSpace.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/MultigridSolver.h \
                          dolfin/StiffnessMatrix.h
INCLUDES                = @MODULE_INCLUDES@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = FiniteElement.cpp \
                          FEM.cpp \
                          FiniteElementMethod.cpp \
                          FunctionSpace.cpp \
                          LoadVector.cpp \
                          MassMatrix.cpp \
                          MultigridSolver.cpp \
                          StiffnessMatrix.cpp
