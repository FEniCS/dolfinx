SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/assemble.h \
                           dolfin/AffineMap.h \
                           dolfin/Assembler.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/BoundaryValue.h \
                           dolfin/OldDofMap.h \
                           dolfin/DofMap.h \
                           dolfin/DofMaps.h \
                           dolfin/UFC.h \
                           dolfin/UFCCell.h \
                           dolfin/UFCMesh.h \
			   dolfin/FEBasis.h \
                           dolfin/FEM.h \
                           dolfin/FiniteElement.h \
                           dolfin/FiniteElementSpec.h \
                           dolfin/NewAffineMap.h \
                           dolfin/P1tet.h \
                           dolfin/P1tri.h \
                           dolfin/DP0tri.h \
                           dolfin/VP1tri.h \
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
libdolfin_fem_la_SOURCES = assemble.cpp \
                           AffineMap.cpp \
                           Assembler.cpp \
                           BoundaryCondition.cpp \
                           BoundaryValue.cpp \
                           OldDofMap.cpp \
                           DofMap.cpp \
                           DofMaps.cpp \
                           UFC.cpp \
			   FEBasis.cpp \
                           FEM.cpp \
                           FiniteElement.cpp \
                           FiniteElementSpec.cpp \
                           NewAffineMap.cpp
libdolfin_fem_la_LDFLAGS = -static
