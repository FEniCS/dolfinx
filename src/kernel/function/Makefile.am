SUBDIRS                       = dolfin
nobase_include_HEADERS        = dolfin/dolfin_function.h \
                                dolfin/ConstantFunction.h \
                                dolfin/DiscreteFunction.h \
                                dolfin/Function.h \
                                dolfin/GenericFunction.h \
                                dolfin/SpecialFunctions.h \
                                dolfin/SubFunction.h \
                                dolfin/UserFunction.h
INCLUDES                      = @KERNEL_CFLAGS@
lib_LTLIBRARIES               = libdolfin-function.la
libdolfin_function_la_SOURCES = ConstantFunction.cpp \
                                DiscreteFunction.cpp \
                                Function.cpp \
                                UserFunction.cpp \
libdolfin_function_la_LDFLAGS = -static
