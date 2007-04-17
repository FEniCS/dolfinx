SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_fem.h \
                           dolfin/assemble.h \
                           dolfin/Assembler.h \
                           dolfin/NewBoundaryCondition.h \
                           dolfin/DofMap.h \
                           dolfin/DofMaps.h \
                           dolfin/Form.h \
                           dolfin/UFC.h \
                           dolfin/UFCCell.h \
                           dolfin/UFCMesh.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-fem.la
libdolfin_fem_la_SOURCES = assemble.cpp \
                           Assembler.cpp \
                           NewBoundaryCondition.cpp \
                           DofMap.cpp \
                           DofMaps.cpp \
                           UFC.cpp
libdolfin_fem_la_LDFLAGS = -static
