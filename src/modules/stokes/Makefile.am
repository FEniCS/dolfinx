SUBDIRS                    = dolfin
nobase_include_HEADERS     = dolfin/dolfin_stokes.h \
                             dolfin/Stokes2D.h \
                             dolfin/Stokes3D.h \
                             dolfin/StokesSolver.h
INCLUDES                   = @MODULE_CFLAGS@
lib_LTLIBRARIES              = libdolfin-stokes.la
libdolfin_stokes_la_SOURCES = StokesSolver.cpp
libdolfin_stokes_la_LDFLAGS = -static
