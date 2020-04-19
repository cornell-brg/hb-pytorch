# Build file for the C10 HammerBlade. It will depend on bsg_manycore.h in the
# future

include(../../cmake/public/utils.cmake)

# ---[ Configure macro file.
set(C10_HAMMERBLADE_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/hammerblade_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/hammerblade/impl/hammerblade_cmake_macros.h)

# Collect all cpp files
file(GLOB C10_HAMMERBLADE_SRCS
        *.cpp
        impl/*.cpp
        )

# Collect all header files
file(GLOB C10_HAMMERBLADE_HEADERS
        *.h
        impl/*.h
        )

message(STATUS "C10_HAMMERBLADE sources: ")
foreach(tmp ${C10_HAMMERBLADE_SRCS})
  message(STATUS "  " ${tmp})
endforeach()

message(STATUS "C10_HAMMERBLADE headers: ")
foreach(tmp ${C10_HAMMERBLADE_HEADERS})
  message(STATUS "  " ${tmp})
endforeach()

add_library(c10_hammerblade ${C10_HAMMERBLADE_SRCS} ${C10_HAMMERBLADE_HEADERS})


# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10_hammerblade PRIVATE "-DC10_HAMMERBLADE_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_hammerblade PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_hammerblade
target_link_libraries(c10_hammerblade PUBLIC c10)

# Dependency to bsg_manycore
if(USE_HB_EMUL)
  # Use BRG CUDALite Emulation
  message( STATUS "Linking CUDALite Emulation Layer...")
  add_subdirectory(emul)
  target_link_libraries(c10_hammerblade PUBLIC bsg_manycore_runtime.1.0)
  target_include_directories( c10_hammerblade PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/emul> )
  target_compile_definitions(c10_hammerblade PRIVATE "-DHB_KERNEL_DIR=NULL")
else()
  # Use real BSG CUDALite
  if( NOT DEFINED ENV{BSG_MANYCORE_DIR} )
    message( FATAL_ERROR "BSG_MANYCORE_DIR not set: Please set BSG_MANYCORE_DIR to bsg_f1/libraries location.")
  endif()
  find_library(BSG_MANYCORE_RUNTIME_LIB bsg_manycore_runtime HINTS $ENV{BSG_MANYCORE_DIR})
  target_link_libraries(c10_hammerblade PUBLIC ${BSG_MANYCORE_RUNTIME_LIB})

  # Dependency on kernel binary path
  if( NOT DEFINED ENV{HB_KERNEL_DIR})
    message( FATAL_ERROR "HB_KERNEL_DIR not set: Please set HB_KERNEL_DIR to kernel binary location.")
  endif()
  target_compile_definitions(c10_hammerblade PRIVATE "-DHB_KERNEL_PATH=$ENV{HB_KERNEL_DIR}")

  target_include_directories( c10_hammerblade PUBLIC $ENV{BSG_MANYCORE_DIR} )
endif()

target_include_directories(
    c10_hammerblade PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# ---[ Installation
install(TARGETS c10_hammerblade EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_HAMMERBLADE_HEADERS})
  get_filename_component( dir ${file} DIRECTORY )
  install( FILES ${file} DESTINATION include/c10/hammerblade/${dir} )
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/hammerblade/impl/hammerblade_cmake_macros.h
  DESTINATION include/c10/hammerblade/impl)

if (MSVC AND C10_HAMMERBLADE_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_hammerblade> DESTINATION lib OPTIONAL)
endif()
