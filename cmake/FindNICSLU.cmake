# NICSLU lib usually requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.

if (NICSLU_INCLUDES AND NICSLU_LIBRARIES)
  set(NICSLU_FIND_QUIETLY TRUE)
endif ()

find_path(NICSLU_INCLUDES
  NAMES
  nicslu.h
  nics_config.h
  PATHS
  $ENV{NICSLUDIR}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  nicslu
  include
  lib
)

find_library(NICSLU_LIBRARIES nicslu PATHS $ENV{NICSLUDIR} ${LIB_INSTALL_DIR})

if(NICSLU_LIBRARIES)

  if(NOT NICSLU_LIBDIR)
    get_filename_component(NICSLU_LIBDIR ${NICSLU_LIBRARIES} PATH)
  endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NICSLU DEFAULT_MSG
                                  NICSLU_INCLUDES NICSLU_LIBRARIES)

mark_as_advanced(NICSLU_INCLUDES NICSLU_LIBRARIES)
