# Build file for the C10 Probe -- our instrumentation library

include(../../cmake/public/utils.cmake)

# ---[ Configure macro file.
set(C10_PROBE_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/probe_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/probe/impl/probe_cmake_macros.h)

# Collect all cpp files
file(GLOB C10_PROBE_SRCS
        *.cpp
        impl/*.cpp
        )

# Collect all header files
file(GLOB C10_PROBE_HEADERS
        *.h
        impl/*.h
        )

message(STATUS "C10_PROBE sources: ")
foreach(tmp ${C10_PROBE_SRCS})
  message(STATUS "  " ${tmp})
endforeach()

message(STATUS "C10_PROBE headers: ")
foreach(tmp ${C10_PROBE_HEADERS})
  message(STATUS "  " ${tmp})
endforeach()

add_library(c10_probe ${C10_PROBE_SRCS} ${C10_PROBE_HEADERS})

# If building with profiler
if(PROFILE_ATEN)
  target_compile_definitions(c10_probe PRIVATE PROFILE_ATEN)
endif()

# Include nlohmann_json
target_include_directories(c10_probe PRIVATE
                           ${CMAKE_SOURCE_DIR}/third_party/include)

# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10_probe PRIVATE "-DC10_PROBE_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_probe PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_probe
target_link_libraries(c10_probe PUBLIC c10)

target_include_directories(
    c10_probe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# ---[ Installation
install(TARGETS c10_probe EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_PROBE_HEADERS})
  get_filename_component( dir ${file} DIRECTORY )
  install( FILES ${file} DESTINATION include/c10/probe/${dir} )
endforeach()
install(FILES ${CMAKE_BINARY_DIR}/c10/probe/impl/probe_cmake_macros.h
  DESTINATION include/c10/probe/impl)

if (MSVC AND C10_PROBE_BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:c10_probe> DESTINATION lib OPTIONAL)
endif()
