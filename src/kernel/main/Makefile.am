SUBDIRS         = dolfin
include_HEADERS = dolfin.h \
                  dolfin/constants.h \
                  dolfin/Problem.h \
                  dolfin/Solver.h
INCLUDES        = @MAIN_CFLAGS@
lib_LIBRARIES   = libdolfin-main.a
libdolfin_main_a_SOURCES = dolfin.h \
                           Problem.cpp \
                           Solver.cpp
