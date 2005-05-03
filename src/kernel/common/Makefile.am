SUBDIRS = dolfin

include_HEADERS = dolfin/dolfin_common.h \
                  dolfin/PArray.h \
                  dolfin/NewArray.h \
                  dolfin/General.h \
                  dolfin/PList.h \
                  dolfin/NewList.h \
                  dolfin/System.h \
                  dolfin/Table.h \
                  dolfin/Tensor.h \
                  dolfin/Variable.h \
                  dolfin/meminfo.h \
                  dolfin/sysinfo.h \
                  dolfin/timeinfo.h \
                  dolfin/utils.h

INCLUDES = @KERNEL_CFLAGS@

lib_LIBRARIES = libdolfin-common.a

libdolfin_common_a_SOURCES = System.cpp \
                             Variable.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
