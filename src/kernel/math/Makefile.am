SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_math.h \
                           dolfin/basic.h \
                           dolfin/Lagrange.h \
                           dolfin/Legendre.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES            = libdolfin-math.la
libdolfin_math_la_SOURCES = basic.cpp \
                           Lagrange.cpp \
                           Legendre.cpp
libdolfin_math_la_LDFLAGS = -static
