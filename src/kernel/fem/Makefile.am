SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/assemble.h \
                           dolfin/Assembler.h \
                           dolfin/BoundaryCondition.h \
                           dolfin/DofMap.h \
                           dolfin/DofMaps.h \
                           dolfin/Form.h \
                           dolfin/SubSystem.h \
                           dolfin/UFC.h \
                           dolfin/UFCCell.h \
                           dolfin/UFCMesh.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = assemble.cpp \
                           Assembler.cpp \
                           BoundaryCondition.cpp \
                           DofMap.cpp \
                           DofMaps.cpp \
                           SubSystem.cpp \
                           UFC.cpp
libdolfin_fem_la_LDFLAGS = -static
