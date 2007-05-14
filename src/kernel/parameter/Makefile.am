SUBDIRS                        = dolfin
nobase_include_HEADERS         = dolfin/dolfin_parameter.h \
                                 dolfin/parameters.h \
                                 dolfin/Parameter.h \
                                 dolfin/ParameterList.h \
                                 dolfin/ParameterSystem.h \
                                 dolfin/ParameterValue.h \
                                 dolfin/Parametrized.h
INCLUDES                       = @KERNEL_CFLAGS@
lib_LTLIBRARIES                = libdolfin-parameter.la
libdolfin_parameter_la_SOURCES = parameters.cpp \
                                 Parameter.cpp \
                                 ParameterList.cpp \
                                 ParameterSystem.cpp \
                                 ParameterValue.cpp \
                                 Parametrized.cpp
libdolfin_parameter_la_LDFLAGS = -static
