install(CODE "MESSAGE(
\"----------------------------------------------------------------------------
DOLFIN has now been installed in

    ${CMAKE_INSTALL_PREFIX}

and the demo programs have been installed in

    ${CMAKE_INSTALL_PREFIX}/${DOLFIN_SHARE_DIR}/demo

Before rushing off to try the demos, don't forget to update your
environment variables. This can be done easily using the helper file
'dolfin.conf' which sets the appropriate variables (for users of the
Bash shell).

To update your environment variables, run the following command:

    source ${CMAKE_INSTALL_PREFIX}/${DOLFIN_SHARE_DIR}/dolfin.conf

For future reference, we recommend that you add this command to your
configuration (.bashrc, .profile or similar).
----------------------------------------------------------------------------\")")
