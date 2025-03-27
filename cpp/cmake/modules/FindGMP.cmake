# - Try to find the GMP libraries
# This module defines:
#  GMP_FOUND             - system has GMP lib
#  GMP_INCLUDE_DIR       - the GMP include directory
#  GMP_LIBRARIES_DIR     - directory where the GMP libraries are located
#  GMP_LIBRARIES         - Link these to use GMP

include(FindPackageHandleStandardArgs)

if(GMP_INCLUDE_DIR)
  set(GMP_in_cache TRUE)
else()
  set(GMP_in_cache FALSE)
endif()
if(NOT GMP_LIBRARIES)
  set(GMP_in_cache FALSE)
endif()

# Is it already configured?
if( NOT GMP_in_cache )

  find_path(GMP_INCLUDE_DIR
            NAMES gmp.h
            HINTS ENV GMP_INC_DIR
                  ENV GMP_DIR
                  $ENV{GMP_DIR}/include
            PATH_SUFFIXES include
  	        DOC "The directory containing the GMP header files"
           )

  find_library(GMP_LIBRARIES NAMES gmp
    HINTS ENV GMP_LIB_DIR
          ENV GMP_DIR
          $ENV{GMP_DIR}/lib
    PATH_SUFFIXES lib
    DOC "Path to the Release GMP library"
    )

    find_library(GMPXX_LIBRARIES NAMES gmpxx
    HINTS ENV GMP_LIB_DIR
          ENV GMP_DIR
          $ENV{GMP_DIR}/lib
    PATH_SUFFIXES lib
    DOC "Path to the Release GMPXX library"
    )
endif()

find_package_handle_standard_args(GMP "DEFAULT_MSG" GMP_LIBRARIES GMP_INCLUDE_DIR)
find_package_handle_standard_args(GMPXX "DEFAULT_MSG" GMPXX_LIBRARIES GMP_INCLUDE_DIR)


if(GMP_FOUND)
add_library(GMP::gmp INTERFACE IMPORTED)
set_property(TARGET GMP::gmp PROPERTY INTERFACE_LINK_LIBRARIES "${GMP_LIBRARIES}")
set_property(TARGET GMP::gmp PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIRS}")
add_library(GMP::gmpxx INTERFACE IMPORTED)
  set_property(TARGET GMP::gmpxx PROPERTY INTERFACE_LINK_LIBRARIES "${GMPXX_LIBRARIES}")
  set_property(TARGET GMP::gmpxx PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIRS}")
endif()
