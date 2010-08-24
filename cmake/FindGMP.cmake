# - Try to find the GMP librairies
# Once done this will define
#
#  GMP_FOUND        - system has GMP lib
#  GMP_INCLUDE_DIRS - include directories for GMP
#  GMP_LIBRARIES    - libraries for GMP

# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (GMP_INCLUDE_DIRS AND GMP_LIBRARIES)
  # Already in cache, be silent
  set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDE_DIRS AND GMP_LIBRARIES)

find_path(GMP_INCLUDE_DIRS
  NAMES gmp.h
  PATHS ${GMP_DIR} $ENV{GMP_DIR}
  PATH_SUFFIXES include
  DOC "Directory where the GMP header file is located"
  )

find_library(GMP_LIBRARIES
  NAMES gmp libgmp
  PATHS ${GMP_DIR} $ENV{GMP_DIR}
  PATH_SUFFIXES lib
  DOC "The GMP libraries"
  )

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GMP DEFAULT_MSG GMP_INCLUDE_DIRS GMP_LIBRARIES)

mark_as_advanced(GMP_INCLUDE_DIRS GMP_LIBRARIES)
