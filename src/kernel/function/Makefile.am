SUBDIRS                      = dolfin
include_HEADERS              = dolfin/dolfin_function.h \
                               dolfin/DofFunction.h \
                               dolfin/ExpressionFunction.h \
                               dolfin/Function.h \
                               dolfin/GenericFunction.h \
                               dolfin/ODEFunction.h \
                               dolfin/ScalarExpressionFunction.h \
                               dolfin/VectorExpressionFunction.h
INCLUDES                     = @MODULE_INCLUDES@
lib_LIBRARIES                = libdolfin-function.a
libdolfin_function_a_SOURCES = DofFunction.cpp \
                               ExpressionFunction.cpp \
                               Function.cpp \
                               GenericFunction.cpp \
                               ODEFunction.cpp \
                               ScalarExpressionFunction.cpp \
                               VectorExpressionFunction.cpp
