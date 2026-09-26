vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jhale/scotch
  REF 03de84453f70ea381a9990cdadd73c82e1f7de36
  SHA512 ba9a59b6ec4bcb73af7ac2ec1ae396247aaff0e792949ddf961bc62b9e6f8259f8b6a7bdc51e9f60f8883f0ed372d585c5f86a00487eada525203571a3c0ba21
  HEAD_REF jhale/windows-fixes-squash
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
          -DLIBPTSCOTCHERR=ptscotcherr
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
