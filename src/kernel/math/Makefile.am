SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_math.h \
                           dolfin/basic.h \
                           dolfin/Lagrange.h \
                           dolfin/Legendre.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LIBRARIES            = libdolfin-math.a
libdolfin_math_a_SOURCES = basic.cpp \
                           Lagrange.cpp \
                           Legendre.cpp
