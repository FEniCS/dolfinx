SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/NewBoundaryCondition.h \
                          dolfin/BoundaryValue.h \
                          dolfin/LinearTetElement.h\
                          dolfin/LinearTriElement.h \
                          dolfin/LoadVector.h \
                          dolfin/MassMatrix.h \
                          dolfin/NewFEM.h \
                          dolfin/NewFiniteElement.h \
                          dolfin/StiffnessMatrix.h
INCLUDES                = @KERNEL_CFLAGS@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = NewBoundaryCondition.cpp \
                          BoundaryValue.cpp \
                          LoadVector.cpp \
                          MassMatrix.cpp \
                          NewFEM.cpp \
                          NewFiniteElement.cpp \
                          StiffnessMatrix.cpp
