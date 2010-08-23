# - Try to find ParMETIS
# Once done this wil define
#
#  PARMETIS_FOUND       - system has ParMETIS
#  PARMETIS_INCLUDE_DIR - include directories for ParMETIS
#  PARMETIS_LIBRARIES   - libraries for ParMETIS

FIND_PATH(PARMETIS_INCLUDE_DIR parmetis.h
  /usr/local/include
  /usr/include
)

FIND_LIBRARY(PARMETIS_LIBRARY parmetis
  /usr/local/lib
  /usr/lib
)

FIND_LIBRARY(METIS_LIBRARY metis
  /usr/local/lib
  /usr/lib
)

IF(PARMETIS_INCLUDE_DIR)
  IF(PARMETIS_LIBRARY)
    SET(PARMETIS_LIBRARIES ${PARMETIS_LIBRARY} ${METIS_LIBRARY})
    SET(PARMETIS_FOUND "YES" )
  ENDIF(PARMETIS_LIBRARY)
ENDIF(PARMETIS_INCLUDE_DIR)

# Standard package handling
find_package_handle_standard_args(ParMETIS.dolfin,
  "ParMETIS could not be found."
  PARMETIS_FOUND)
