install(CODE "MESSAGE(
\"------------------------------------------------------------------------------
DOLFIN has been installed in

    ${CMAKE_INSTALL_PREFIX}

and the demo programs have been installed in

    ${CMAKE_INSTALL_PREFIX}/${DOLFIN_SHARE_DIR}/demo

A helper file is provided for BASH users to set the appropriate environment
variables. Before compiling and running programs, source the helper file
'dolfin.conf':

    source ${CMAKE_INSTALL_PREFIX}/${DOLFIN_SHARE_DIR}/dolfin.conf
------------------------------------------------------------------------------\")")
