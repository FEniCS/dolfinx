SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_fem.h \
                          dolfin/ElementFunction.h \
                          dolfin/Equation.h \
                          dolfin/FiniteElement.h \
                          dolfin/Function.h \
                          dolfin/FunctionList.h \
                          dolfin/FunctionSpace.h \
                          dolfin/Galerkin.h \
                          dolfin/Product.h \
                          dolfin/ShapeFunction.h \
                          dolfin/Solver.h \
                          dolfin/TetLinMapping.h \
                          dolfin/TriLinMapping.h
INCLUDES                = @MODULE_INCLUDES@
lib_LIBRARIES           = libdolfin-fem.a
libdolfin_fem_a_SOURCES = ElementFunction.cpp \
                          Equation.cpp \
                          EquationSystem.h \
                          FiniteElement.cpp \
                          Function.cpp \
                          FunctionList.cpp \
                          FunctionSpace.cpp \
                          Galerkin.cpp \
                          Integral.cpp \
                          Mapping.cpp \
                          Product.cpp \
                          ShapeFunction.cpp \
                          TetLinMapping.cpp \
                          TriLinMapping.cpp
