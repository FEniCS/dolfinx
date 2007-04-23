SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin.h \
                           dolfin/dolfin_main.h \
                           dolfin/constants.h \
                           dolfin/init.h 
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES            = libdolfin-main.la
libdolfin_main_la_SOURCES = dolfin.h \
                           init.cpp 
libdolfin_main_la_LDFLAGS = -static
