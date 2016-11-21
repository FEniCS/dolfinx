set(DIR parameter)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_parameter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalParameters.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Parameter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Parameters.h
  )

# Install headers
install(FILES ${DOLFIN_${DIR}_HEADERS}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR}
  COMPONENT Development)

# Append headers
set(DOLFIN_HEADERS ${DOLFIN_HEADERS} ${DOLFIN_${DIR}_HEADERS} PARENT_SCOPE)

# Append sources
set(DOLFIN_SOURCES
  ${DOLFIN_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalParameters.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Parameter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Parameters.cpp
  PARENT_SCOPE)
