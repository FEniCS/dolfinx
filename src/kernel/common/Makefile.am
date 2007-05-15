SUBDIRS = dolfin

nobase_include_HEADERS = dolfin/dolfin_common.h \
                         dolfin/Array.h \
                         dolfin/List.h \
                         dolfin/TimeDependent.h \
                         dolfin/Variable.h \
                         dolfin/timing.h \
                         dolfin/unittest.h \
                         dolfin/utils.h

INCLUDES = @KERNEL_CFLAGS@

lib_LTLIBRARIES = libdolfin-common.la

libdolfin_common_la_SOURCES = TimeDependent.cpp \
			      Variable.cpp \
                              timing.cpp \
                              utils.cpp
libdolfin_common_la_LDFLAGS = -static
