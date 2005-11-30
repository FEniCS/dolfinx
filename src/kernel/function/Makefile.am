SUBDIRS                      = dolfin
nobase_include_HEADERS       = dolfin/dolfin_function.h \
                               dolfin/DiscreteFunction.h \
                               dolfin/Function.h \
                               dolfin/FunctionPointer.h \
                               dolfin/FunctionPointerFunction.h \
                               dolfin/GenericFunction.h \
                               dolfin/LocalFunctionData.h \
                               dolfin/UserFunction.h
INCLUDES                     = @KERNEL_CFLAGS@
lib_LIBRARIES                = libdolfin-function.a
libdolfin_function_a_SOURCES = DiscreteFunction.cpp \
                               Function.cpp \
                               FunctionPointerFunction.cpp \
                               GenericFunction.cpp \
                               LocalFunctionData.cpp \
                               UserFunction.cpp
