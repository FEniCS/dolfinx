vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
vcpkg_from_gitlab(
  GITLAB_URL https://gitlab.inria.fr
  OUT_SOURCE_PATH SOURCE_PATH
  REPO scotch/scotch
  REF "v${VERSION}"
  SHA512 dbb41f82b1113f7e109c600ace9dbcf82fec5cff8944d92ed2f62917aef3c619b580421bb847b7108197ce6139302f7af0d4088b0447710f693606608b60ed5f
  HEAD_REF master
)

vcpkg_find_acquire_program(FLEX)
cmake_path(GET FLEX PARENT_PATH FLEX_DIR)
vcpkg_add_to_path("${FLEX_DIR}")

vcpkg_find_acquire_program(BISON)
cmake_path(GET BISON PARENT_PATH BISON_DIR)
vcpkg_add_to_path("${BISON_DIR}")

# Uses gcc intrinsics otherwise
string(APPEND VCPKG_C_FLAGS " -DGRAPHMATCHNOTHREAD")
string(APPEND VCPKG_CXX_FLAGS " -DGRAPHMATCHNOTHREAD")

vcpkg_check_features(
  OUT_FEATURE_OPTIONS
  FEATURE_OPTIONS
  FEATURES
  ptscotch
  BUILD_PTSCOTCH
  metis
  BUILD_LIBSCOTCHMETIS
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS ${FEATURE_OPTIONS} -DBUILD_LIBESMUMPS=OFF -DBUILD_FORTRAN=OFF -DTHREADS=ON
          -DMPI_THREAD_MULTIPLE=OFF -DINSTALL_METIS_HEADERS=ON -DLIBSCOTCHERR=scotcherr
          -DLIBPTSCOTCHERR=ptscotcherr -DENABLE_TESTS=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/scotch")
vcpkg_copy_tools(
  TOOL_NAMES
  acpl
  amk_ccc
  amk_fft2
  amk_grf
  amk_hy
  amk_m2
  amk_p2
  atst
  gbase
  gcv
  gmap
  gmk_hy
  gmk_m2
  gmk_m3
  gmk_msh
  gmk_ub2
  gmtst
  gord
  gotst
  gscat
  gtst
  mcv
  mmk_m2
  mmk_m3
  mord
  mtst
  AUTO_CLEAN
)

if("ptscotch" IN_LIST FEATURES)
  vcpkg_copy_tools(TOOL_NAMES dggath dgmap dgord dgscat dgtst AUTO_CLEAN)
  if("metis" IN_LIST FEATURES)
    # adm2dgr is only built when both PT-Scotch and the ScotchMeTiS
    # compatibility library are enabled, since it links against
    # libPTScotchParMeTiS
    vcpkg_copy_tools(TOOL_NAMES adm2dgr AUTO_CLEAN)
  endif()
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/doc/CeCILL-C_V1-en.txt")

file(
  REMOVE_RECURSE
  "${CURRENT_PACKAGES_DIR}/debug/include"
  "${CURRENT_PACKAGES_DIR}/debug/man"
  "${CURRENT_PACKAGES_DIR}/man"
  "${CURRENT_PACKAGES_DIR}/debug/share"
)
