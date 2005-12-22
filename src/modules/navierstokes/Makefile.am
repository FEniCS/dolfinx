SUBDIRS                          = dolfin
nobase_include_HEADERS           = dolfin/dolfin_navierstokes.h \
                                   dolfin/NSEMomentum.h \
                                   dolfin/NSEContinuity.h \
                                   dolfin/NSESolver.h
INCLUDES                         = @MODULE_CFLAGS@
lib_LIBRARIES                    = libdolfin-navierstokes.a
libdolfin_navierstokes_a_SOURCES = NSESolver.cpp

# Temporary fix to speed up compilation of form
#CXXFLAGS = '-DDEBUG=1 -g -O0 -Wall -Werror -ansi -std=c++98'
