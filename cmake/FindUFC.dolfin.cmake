# Try to find UFC - the unified framework for finite element assembly
#
# This module defines the following variables:
#
#   UFC_FOUND        - system has UFC
#   UFC_INCLUDE_DIRS - UFC include directory
#   UFC_VERSION      - UFC version string (MAJOR.MINOR.MICRO)

include(FindPkgConfig)
pkg_check_modules(UFC REQUIRED ufc-1>=1.4.1)
