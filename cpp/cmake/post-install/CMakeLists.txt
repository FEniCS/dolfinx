install(
  CODE "MESSAGE(
\"----------------------------------------------------------------------------
DOLFINx has now been installed in

    ${CMAKE_INSTALL_PREFIX}

and demo programs have been installed in

    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/dolfinx/demo

Don't forget to update your environment variables. This can be done
easily using the helper file 'dolfinx.conf' which sets the appropriate
variables (for users of the Bash shell).

To update your environment variables, run the following command:

    source ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/dolfinx/dolfinx.conf

----------------------------------------------------------------------------\")"
)
