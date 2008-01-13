SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/assemble.h \
                           dolfin/Assembler.h \
                           dolfin/pAssembler.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/DirichletBC.h \
                           dolfin/DofMap.h \
			   dolfin/pDofMap.h \
                           dolfin/DofMapSet.h \
			   dolfin/pDofMapSet.h \
                           dolfin/Form.h \
                           dolfin/pForm.h \
                           dolfin/PeriodicBC.h \
                           dolfin/SparsityPatternBuilder.h \
                           dolfin/pSparsityPatternBuilder.h \
                           dolfin/SubSystem.h \
                           dolfin/UFC.h \
                           dolfin/pUFC.h \
                           dolfin/UFCCell.h \
                           dolfin/UFCMesh.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = assemble.cpp \
                           Assembler.cpp \
                           pAssembler.cpp \
                           BoundaryCondition.cpp \
                           DirichletBC.cpp \
                           DofMap.cpp \
			   pDofMap.cpp \
                           DofMapSet.cpp \
			   pDofMapSet.cpp \
                           Form.cpp \
                           pForm.cpp \
                           PeriodicBC.cpp \
                           SparsityPatternBuilder.cpp \
                           pSparsityPatternBuilder.cpp \
                           SubSystem.cpp \
                           UFC.cpp \
                           pUFC.cpp
libdolfin_fem_la_LDFLAGS = -static
