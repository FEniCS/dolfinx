SUBDIRS = dolfin

nobase_include_HEADERS = dolfin/dolfin_common.h \
                         dolfin/Array.h \
                         dolfin/List.h \
                         dolfin/PArray.h \
                         dolfin/PList.h \
                         dolfin/Table.h \
                         dolfin/Tensor.h \
                         dolfin/TimeDependent.h \
                         dolfin/Variable.h \
                         dolfin/meminfo.h \
                         dolfin/timeinfo.h \
                         dolfin/utils.h

INCLUDES = @KERNEL_CFLAGS@

lib_LIBRARIES = libdolfin-common.a

libdolfin_common_a_SOURCES = TimeDependent.cpp \
			     Variable.cpp \
                             meminfo.cpp \
                             timeinfo.cpp \
                             utils.cpp
