install(
  CODE
    "MESSAGE(
\"----------------------------------------------------------------------------
DOLFINx has now been installed in

    ${CMAKE_INSTALL_PREFIX}

and demo programs have been installed in

    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/dolfinx/demo

----------------------------------------------------------------------------\")"
)
