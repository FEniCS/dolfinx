SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/BoundaryCondition.h \
                          dolfin/DxMatrix.h \
                          dolfin/DyMatrix.h \
                          dolfin/DzMatrix.h \
                          dolfin/FEM.h \
                          dolfin/FiniteElement.h \
                          dolfin/FiniteElementMethod.h \
                          dolfin/FunctionSpace.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/MultigridSolver.h \
                          dolfin/NewFEM.h \
                          dolfin/NewFiniteElement.h \
                          dolfin/StiffnessMatrix.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = DxMatrix.cpp \
                          DyMatrix.cpp \
                          DzMatrix.cpp \
                          FiniteElement.cpp \
                          FEM.cpp \
                          FiniteElementMethod.cpp \
                          FunctionSpace.cpp \
                          LoadVector.cpp \
                          MassMatrix.cpp \
                          MultigridSolver.cpp \
                          NewFEM.cpp \
                          NewFiniteElement.cpp \
                          StiffnessMatrix.cpp
