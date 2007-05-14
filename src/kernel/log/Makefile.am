SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_log.h \
                           dolfin/log.h \
                           dolfin/Event.h \
                           dolfin/Logger.h \
                           dolfin/LogManager.h \
                           dolfin/LogStream.h \
                           dolfin/Progress.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-log.la
libdolfin_log_la_SOURCES = log.cpp \
                           Event.cpp \
                           Logger.cpp \
                           LogManager.cpp \
                           LogStream.cpp \
                           Progress.cpp
libdolfin_log_la_LDFLAGS = -static
