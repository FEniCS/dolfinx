SUBDIRS                       = dolfin
nobase_include_HEADERS        = dolfin/dolfin_elements.h dolfin/ElementLibrary.h
INCLUDES                      = @KERNEL_CFLAGS@
lib_LTLIBRARIES               = libdolfin-elements.la
libdolfin_elements_la_SOURCES = ElementLibrary.cpp
libdolfin_elements_la_LDFLAGS = -static
