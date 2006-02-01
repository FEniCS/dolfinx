SUBDIRS                          = dolfin
nobase_include_HEADERS           = dolfin/dolfin_navierstokes.h \
                                   dolfin/NSEMomentum2D.h \
                                   dolfin/NSEContinuity2D.h \
                                   dolfin/NSEMomentum3D.h \
                                   dolfin/NSEContinuity3D.h \
                                   dolfin/NSESolver.h
INCLUDES                         = @MODULE_CFLAGS@
lib_LTLIBRARIES                    = libdolfin-navierstokes.la
libdolfin_navierstokes_la_SOURCES = NSESolver.cpp
libdolfin_navierstokes_la_LDFLAGS = -static

# Temporary fix to speed up compilation of form
CXXFLAGS = '-DDEBUG=1 -g -O0 -Wall -Werror -ansi -std=c++98'

