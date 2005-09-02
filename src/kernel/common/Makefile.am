SUBDIRS = dolfin

include_HEADERS = dolfin/dolfin_common.h \
                  dolfin/Array.h \
                  dolfin/List.h \
                  dolfin/PArray.h \
                  dolfin/PList.h \
                  dolfin/Synchronizer.h \
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

libdolfin_common_a_SOURCES = Synchronizer.cpp \
                             System.cpp \
                             Variable.cpp \
                             meminfo.cpp \
                             sysinfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
