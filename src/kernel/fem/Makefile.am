SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/BoundaryCondition.h \
                          dolfin/DofFunction.h \
                          dolfin/ExpressionFunction.h \
                          dolfin/FiniteElement.h \
                          dolfin/Function.h \
                          dolfin/FunctionSpace.h \
                          dolfin/Galerkin.h \
                          dolfin/GenericFunction.h \
                          dolfin/ScalarExpressionFunction.h \
                          dolfin/VectorExpressionFunction.h
INCLUDES                = @MODULE_INCLUDES@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = DofFunction.cpp \
                          ExpressionFunction.cpp \
                          FiniteElement.cpp \
                          Function.cpp \
                          FunctionSpace.cpp \
                          GenericFunction.cpp \
                          Galerkin.cpp \
                          ScalarExpressionFunction.cpp \
                          VectorExpressionFunction.cpp
