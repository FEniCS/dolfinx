SUBDIRS                 = dolfin
include_HEADERS         = dolfin/dolfin_log.h \
                          dolfin/GenericLogger.h \
                          dolfin/Logger.h \
                          dolfin/LoggerMacros.h \
                          dolfin/LogManager.h \
                          dolfin/LogStream.h \
                          dolfin/TerminalLogger.h
INCLUDES                = @MODULE_INCLUDES@
lib_LIBRARIES           = libdolfin-log.a
libdolfin_log_a_SOURCES = GenericLogger.cpp \
                          Logger.cpp \
                          LoggerMacros.cpp \
                          LogManager.cpp \
                          LogStream.cpp \
                          TerminalLogger.cpp
