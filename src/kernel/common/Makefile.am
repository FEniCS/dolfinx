SUBDIRS               = dolfin
include_HEADERS       = dolfin/dolfin_common.h \
                        dolfin/List.h \
                        dolfin/function.h \
                        dolfin/Settings.h \
                        dolfin/System.h \
                        dolfin/Parameter.h \
                        dolfin/ParameterList.h \
                        dolfin/ShortList.h \
                        dolfin/Tensor.h \
                        dolfin/Variable.h \
                        dolfin/meminfo.h \
                        dolfin/sysinfo.h \
                        dolfin/timeinfo.h \
                        dolfin/utils.h
INCLUDES              = @MODULE_INCLUDES@
lib_LIBRARIES         = libdolfin-common.a
libdolfin_common_a_SOURCES = Parameter.cpp \
                             ParameterList.cpp \
                             Settings.cpp \
                             System.cpp \
                             Variable.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
