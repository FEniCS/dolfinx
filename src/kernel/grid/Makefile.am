SUBDIRS                  = dolfin
include_HEADERS          = dolfin/dolfin_grid.h \
                           dolfin/Grid.h \
                           dolfin/Node.h \
                           dolfin/NodeIterator.h \
                           dolfin/Point.h \
                           dolfin/Cell.h \
                           dolfin/CellIterator.h \
                           dolfin/GenericCell.h \
                           dolfin/Triangle.h \
                           dolfin/Tetrahedron.h \
                           dolfin/Edge.h
INCLUDES                 = @MODULE_INCLUDES@
lib_LIBRARIES            = libdolfin-grid.a
libdolfin_grid_a_SOURCES = Cell.cpp \
                           CellIterator.cpp \
                           Grid.cpp \
                           GridData.h \
                           GridData.cpp \
                           InitGrid.h \
                           InitGrid.cpp \
                           Node.cpp \
                           NodeIterator.cpp \
                           Point.cpp \
                           Tetrahedron.cpp \
                           Triangle.cpp
