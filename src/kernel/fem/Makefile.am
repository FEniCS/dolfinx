SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/AffineMap.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/BoundaryValue.h \
                           dolfin/FEM.h \
                           dolfin/FiniteElement.h \
                           dolfin/FiniteElementSpec.h \
                           dolfin/P1tet.h \
                           dolfin/P1tri.h \
                           dolfin/P2tet.h \
                           dolfin/P2tri.h \
                           dolfin/P3tet.h \
                           dolfin/P3tri.h \
                           dolfin/P4tet.h \
                           dolfin/P4tri.h \
                           dolfin/P5tet.h \
                           dolfin/P5tri.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = AffineMap.cpp \
                           BoundaryCondition.cpp \
                           BoundaryValue.cpp \
                           FEM.cpp \
                           FiniteElement.cpp \
                           FiniteElementSpec.cpp
libdolfin_fem_la_LDFLAGS = -static
