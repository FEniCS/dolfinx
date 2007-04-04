SUBDIRS                       = dolfin
nobase_include_HEADERS        = dolfin/dolfin_function.h \
                                dolfin/ConstantFunction.h \
                                dolfin/Function.h \
                                dolfin/FunctionPointerFunction.h \
                                dolfin/FunctionPointer.h \
                                dolfin/GenericFunction.h \
                                dolfin/DiscreteFunction.h \
                                dolfin/SpecialFunctions.h \
                                dolfin/UserFunction.h
INCLUDES                      = @KERNEL_CFLAGS@
lib_LTLIBRARIES               = libdolfin-function.la
libdolfin_function_la_SOURCES = ConstantFunction.cpp \
                                Function.cpp \
                                FunctionPointerFunction.cpp \
                                DiscreteFunction.cpp \
                                UserFunction.cpp \
libdolfin_function_la_LDFLAGS = -static
