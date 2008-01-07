SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin.h \
                           dolfin/dolfin_main.h \
                           dolfin/constants.h \
                           dolfin/init.h \
                           dolfin/MPI.h \
                           dolfin/SubSystemsManager.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES           = libdolfin-main.la
libdolfin_main_la_SOURCES = dolfin.h \
                            init.cpp \
                            MPI.cpp \
                            SubSystemsManager.cpp
libdolfin_main_la_LDFLAGS = -static
