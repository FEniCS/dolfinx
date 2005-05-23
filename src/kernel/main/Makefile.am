SUBDIRS         = dolfin
include_HEADERS = dolfin.h \
                  dolfin/dolfin_main.h \
                  dolfin/constants.h \
                  dolfin/init.h 
#                  dolfin/Problem.h \
#                  dolfin/Solver.h
INCLUDES        = @KERNEL_CFLAGS@ @MODULE_CFLAGS@
lib_LIBRARIES   = libdolfin-main.a
libdolfin_main_a_SOURCES = dolfin.h \
                           init.cpp 
#                           Problem.cpp \
#                           Solver.cpp
