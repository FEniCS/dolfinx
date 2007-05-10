SUBDIRS                  = dolfin
nobase_include_HEADERS   = dolfin/dolfin_log.h \
                           dolfin/log.h \
                           dolfin/Buffer.h \
                           dolfin/CursesLogger.h \
                           dolfin/Event.h \
                           dolfin/GenericLogger.h \
                           dolfin/Logger.h \
                           dolfin/LogManager.h \
                           dolfin/LogStream.h \
                           dolfin/Progress.h \
                           dolfin/SilentLogger.h \
                           dolfin/TerminalLogger.h
INCLUDES                 = @KERNEL_CFLAGS@
lib_LTLIBRARIES          = libdolfin-log.la
libdolfin_log_la_SOURCES = log.cpp \
                           Buffer.cpp \
                           GenericLogger.cpp \
                           CursesLogger.cpp \
                           Event.cpp \
                           Logger.cpp \
                           LogManager.cpp \
                           LogStream.cpp \
                           Progress.cpp \
                           SilentLogger.cpp \
                           TerminalLogger.cpp
libdolfin_log_la_LDFLAGS = -static
