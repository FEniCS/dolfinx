SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/assemble.h \
                           dolfin/Assembler.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/DirichletBC.h \
                           dolfin/DofMap.h \
                           dolfin/DofMapSet.h \
                           dolfin/Form.h \
                           dolfin/PeriodicBC.h \
                           dolfin/SparsityPatternBuilder.h \
                           dolfin/SubSystem.h \
                           dolfin/UFC.h \
                           dolfin/UFCCell.h \
                           dolfin/UFCMesh.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = assemble.cpp \
                           Assembler.cpp \
                           BoundaryCondition.cpp \
                           DirichletBC.cpp \
                           DofMap.cpp \
                           DofMapSet.cpp \
                           PeriodicBC.cpp \
                           SparsityPatternBuilder.cpp \
                           SubSystem.cpp \
                           UFC.cpp
libdolfin_fem_la_LDFLAGS = -static
