SUBDIRS                       = dolfin
nobase_include_HEADERS        = dolfin/dolfin_function.h \
                                dolfin/DiscreteFunction.h \
                                dolfin/Function.h \
                                dolfin/FunctionPointer.h \
                                dolfin/FunctionPointerFunction.h \
                                dolfin/GenericFunction.h \
                                dolfin/LocalFunctionData.h \
                                dolfin/SpecialFunctions.h \
                                dolfin/UserFunction.h
INCLUDES                      = @KERNEL_CFLAGS@
lib_LTLIBRARIES               = libdolfin-function.la
libdolfin_function_la_SOURCES = DiscreteFunction.cpp \
                                Function.cpp \
                                FunctionPointerFunction.cpp \
                                GenericFunction.cpp \
                                LocalFunctionData.cpp \
                                UserFunction.cpp
libdolfin_function_la_LDFLAGS = -static
