SUBDIRS                      = dolfin
nobase_include_HEADERS       = dolfin/dolfin_convdiff.h \
                               dolfin/ConvectionDiffusion2D.h \
                               dolfin/ConvectionDiffusion3D.h \
                               dolfin/ConvectionDiffusionSolver.h
INCLUDES                     = @MODULE_CFLAGS@
lib_LTLIBRARIES                = libdolfin-convdiff.la
libdolfin_convdiff_la_SOURCES = ConvectionDiffusionSolver.cpp
libdolfin_convdiff_la_LDFLAGS = -static
