SUBDIRS                  = dolfin
include_HEADERS          = dolfin/dolfin_grid.h \
                           dolfin/Grid.h \
                           dolfin/GridIterators.h \
                           dolfin/Node.h \
                           dolfin/Point.h \
                           dolfin/Cell.h \
                           dolfin/GenericCell.h \
                           dolfin/Triangle.h \
                           dolfin/Tetrahedron.h \
                           dolfin/Edge.h
INCLUDES                 = @MODULE_INCLUDES@
lib_LIBRARIES            = libdolfin-grid.a
include_HEADERS          = 
libdolfin_grid_a_SOURCES = Cell.cpp \
                           GenericCell.cpp \
                           Grid.cpp \
                           GridData.h \
                           GridData.cpp \
                           GridIterators.cpp \
                           InitGrid.h \
                           InitGrid.cpp \
                           Node.cpp \
                           Point.cpp \
                           Tetrahedron.cpp \
                           Triangle.cpp
