# - Try to find UFC
# Once done this will define
#
#   UFC_FOUND        - system has UFC
#   UFC_INCLUDE_DIRS - include directories for UFC
#   UFC_VERSION      - UFC version string (MAJOR.MINOR.MICRO)

# FIXME: REQUIRED should not be set here

include(FindPkgConfig)
pkg_check_modules(UFC REQUIRED ufc-1>=1.4.1)
