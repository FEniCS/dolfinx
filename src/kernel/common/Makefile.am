SUBDIRS               = dolfin
include_HEADERS       = dolfin/List.h \
                        dolfin/function.h \
                        dolfin/Settings.h \
                        dolfin/System.h \
                        dolfin/Parameter.h \
                        dolfin/ParameterList.h \
                        dolfin/meminfo.h \
                        dolfin/sysinfo.h \
                        dolfin/timeinfo.h \
                        dolfin/utils.h
INCLUDES              = @MODULE_INCLUDES@
lib_LIBRARIES         = libdolfin-common.a
libdolfin_common_a_SOURCES = utils.cpp \
                             Parameter.cpp \
                             ParameterList.cpp \
                             System.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp
