# - Try to find the LibXml++ xml processing library
# Once done this will define
#
#  LIBXMLPP_FOUND - System has LibXml++
#  LIBXMLPP_INCLUDE_DIR - The LibXml++ include directory
#  LIBXMLPP_LIBRARIES - The libraries needed to use LibXml++
#  LIBXMLPP_DEFINITIONS - Compiler switches required for using LibXml++

#=============================================================================
# Copyright 2006-2009 Kitware, Inc.
# Copyright 2006 Alexander Neundorf <neundorf@kde.org>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

# use pkg-config to get the directories and then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls
FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(PC_LIBXML libxml++-2.6 QUIET)
SET(LIBXMLPP_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})

FIND_PATH(LIBXMLPP_INCLUDE_DIR NAMES libxml++/libxml++.h
   HINTS
   ${PC_LIBXML_INCLUDEDIR}
   ${PC_LIBXML_INCLUDE_DIRS}
   PATH_SUFFIXES libxml++-2.6
   )

FIND_LIBRARY(LIBXMLPP_LIBRARIES NAMES xml++-2.6 libxml++-2.6
   HINTS
   ${PC_LIBXML_LIBDIR}
   ${PC_LIBXML_LIBRARY_DIRS}
   )

#FIND_PROGRAM(LIBXML2_XMLLINT_EXECUTABLE xmllint)
## for backwards compat. with KDE 4.0.x:
#SET(XMLLINT_EXECUTABLE "${LIBXML2_XMLLINT_EXECUTABLE}")

# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE if
# all listed variables are TRUE
#INCLUDE("${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake")
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibXml++ DEFAULT_MSG LIBXMLPP_LIBRARIES LIBXMLPP_INCLUDE_DIR)

MARK_AS_ADVANCED(LIBXMLPP_INCLUDE_DIR LIBXMLPP_LIBRARIES)

