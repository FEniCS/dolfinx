SUBDIRS = dolfin

include_HEADERS = dolfin/dolfin_common.h \
                  dolfin/Array.h \
                  dolfin/List.h \
                  dolfin/System.h \
                  dolfin/Table.h \
                  dolfin/Tensor.h \
                  dolfin/Variable.h \
                  dolfin/bcfunction.h \
                  dolfin/meminfo.h \
                  dolfin/function.h \
                  dolfin/sysinfo.h \
                  dolfin/timeinfo.h \
                  dolfin/utils.h \
                  dolfin/vfunction.h

INCLUDES = @MODULE_INCLUDES@

lib_LIBRARIES = libdolfin-common.a

libdolfin_common_a_SOURCES = System.cpp \
                             Variable.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
