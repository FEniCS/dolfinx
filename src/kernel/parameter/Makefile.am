SUBDIRS                       = dolfin
nobase_include_HEADERS        = dolfin/dolfin_parameter.h \
                                dolfin/Parameter.h \
                                dolfin/ParameterList.h \
                                dolfin/ParameterSystem.h \
                                dolfin/ParameterValue.h
INCLUDES                      = @KERNEL_CFLAGS@
lib_LIBRARIES                 = libdolfin-parameter.a
libdolfin_parameter_a_SOURCES = Parameter.cpp \
                                ParameterList.cpp \
                                ParameterSystem.cpp \
                                ParameterValue.cpp
