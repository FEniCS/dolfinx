SUBDIRS                     = dolfin
include_HEADERS             = dolfin/dolfin_element.h \
                              dolfin/P1Tri.h \
                              dolfin/P1Tet.h \
                              dolfin/P1TriElement.h \
                              dolfin/P1TetElement.h \
                              dolfin/shapefunctions.h
INCLUDES                    = @KERNEL_CFLAGS@
lib_LIBRARIES               = libdolfin-element.a
libdolfin_element_a_SOURCES = shapefunctions.cpp
