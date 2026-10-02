cmake_minimum_required(VERSION 3.22.1)
project(freshness_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# android/app/src/main/cpp -> root project
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)

set(CROSSCHICK_BUILD_CLI OFF CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_ROOT}/inference_cpp" "${CMAKE_CURRENT_BINARY_DIR}/inference_cpp")

add_library(freshness_jni SHARED freshness_jni.cpp)
target_link_libraries(freshness_jni PRIVATE crosschick_inference jnigraphics android log)
target_link_options(freshness_jni PRIVATE
    -Wl,--gc-sections
    -Wl,-z,max-page-size=16384)  # dukungan perangkat 16 KB page size (Android 15+)
