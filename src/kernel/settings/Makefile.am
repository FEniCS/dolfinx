SUBDIRS                      = dolfin
nobase_include_HEADERS       = dolfin/dolfin_settings.h \
                               dolfin/Parameter.h \
                               dolfin/ParameterList.h \
                               dolfin/Settings.h \
                               dolfin/SettingsMacros.h \
                               dolfin/SettingsManager.h
INCLUDES                     = @KERNEL_CFLAGS@
lib_LIBRARIES                = libdolfin-settings.a
libdolfin_settings_a_SOURCES = Parameter.cpp \
                               ParameterList.cpp \
                               SettingsMacros.cpp \
                               SettingsManager.cpp
