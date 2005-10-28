SUBDIRS                 = dolfin
nobase_include_HEADERS  = dolfin/dolfin_fem.h \
                          dolfin/AffineMap.h \
                          dolfin/BoundaryCondition.h \
                          dolfin/BoundaryValue.h \
                          dolfin/FEM.h \
                          dolfin/FiniteElement.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/P1Tet.h \
                          dolfin/P1Tri.h \
                          dolfin/StiffnessMatrix.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = AffineMap.cpp \
                          BoundaryCondition.cpp \
                          BoundaryValue.cpp \
                          FEM.cpp \
                          FiniteElement.cpp \
                          LoadVector.cpp \
                          MassMatrix.cpp \
                          StiffnessMatrix.cpp
