SUBDIRS                     = dolfin
nobase_include_HEADERS      = dolfin/dolfin_fsi.h \
                              dolfin/FSIContinuity2D.h \
                              dolfin/FSIContinuity3D.h \
                              dolfin/FSIDotSigma2D.h \
                              dolfin/FSIDotSigma3D.h \
                              dolfin/FSIMomentum2D.h \
                              dolfin/FSIMomentum3D.h \
                              dolfin/FSISolver.h
INCLUDES                    = @MODULE_CFLAGS@
lib_LTLIBRARIES               = libdolfin-fsi.la
libdolfin_fsi_la_SOURCES = FSISolver.cpp
libdolfin_fsi_la_LDFLAGS = -static
