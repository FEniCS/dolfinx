SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin.h \
                           dolfin/dolfin_main.h \
                           dolfin/constants.h \
                           dolfin/init.h 
#                  dolfin/Problem.h \
#                  dolfin/Solver.h
INCLUDES                 = @KERNEL_CFLAGS@ @SOLVER_CFLAGS@
lib_LTLIBRARIES            = libdolfin-main.la
libdolfin_main_la_SOURCES = dolfin.h \
                           init.cpp 
#                           Problem.cpp \
#                           Solver.cpp
libdolfin_main_la_LDFLAGS = -static
