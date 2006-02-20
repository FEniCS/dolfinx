SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/AffineMap.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/BoundaryValue.h \
                           dolfin/FEM.h \
                           dolfin/FiniteElement.h \
                           dolfin/FiniteElementSpec.h \
                           dolfin/LoadVector.h \
                           dolfin/MassMatrix.h \
                           dolfin/P1Tet.h \
                           dolfin/P1TetTemplate.h \
                           dolfin/P1Tri.h \
                           dolfin/P1TriTemplate.h \
                           dolfin/StiffnessMatrix.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = AffineMap.cpp \
                           BoundaryCondition.cpp \
                           BoundaryValue.cpp \
                           FEM.cpp \
                           FiniteElement.cpp \
                           FiniteElementSpec.cpp \
                           LoadVector.cpp \
                           MassMatrix.cpp \
                           StiffnessMatrix.cpp
libdolfin_fem_la_LDFLAGS = -static
