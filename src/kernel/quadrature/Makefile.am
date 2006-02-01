SUBDIRS                        = dolfin
nobase_include_HEADERS         = dolfin/dolfin_quadrature.h \
                                 dolfin/GaussianQuadrature.h \
                                 dolfin/GaussQuadrature.h \
                                 dolfin/LobattoQuadrature.h \
                                 dolfin/Quadrature.h \
                                 dolfin/RadauQuadrature.h
INCLUDES                       = @KERNEL_CFLAGS@
lib_LTLIBRARIES                  = libdolfin-quadrature.la
libdolfin_quadrature_la_SOURCES = GaussianQuadrature.cpp \
                                 GaussQuadrature.cpp \
                                 LobattoQuadrature.cpp \
                                 Quadrature.cpp \
                                 RadauQuadrature.cpp
libdolfin_quadrature_la_LDFLAGS = -static
