cmake_minimum_required(VERSION 3.16)
project(crosschick_inference LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/tensorflow"
    CACHE PATH "Root source TensorFlow (hasil scripts/setup_tensorflow_source)")
option(CROSSCHICK_BUILD_CLI "Build CLI classify_image (desktop)" ON)

# ---- TensorFlow Lite (LiteRT) C++ dibangun dari source resmi Google ----
if(NOT TARGET tensorflow-lite)
  if(NOT EXISTS "${TFLITE_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
    message(FATAL_ERROR
      "Source TensorFlow tidak ditemukan di ${TFLITE_SOURCE_DIR}.\n"
      "Jalankan scripts/setup_tensorflow_source.ps1 (Windows) atau .sh (Linux/macOS).")
  endif()
  set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
  set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
  add_subdirectory("${TFLITE_SOURCE_DIR}/tensorflow/lite"
                   "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
endif()

# ---- Library inti (dipakai CLI desktop & JNI Android) ----
add_library(crosschick_inference STATIC src/freshness_classifier.cpp)
target_include_directories(crosschick_inference PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(crosschick_inference PUBLIC tensorflow-lite)

# ---- CLI desktop ----
if(CROSSCHICK_BUILD_CLI AND NOT ANDROID)
  # Lokasi tetap agar bisa dipakai includePath IntelliSense (.vscode/c_cpp_properties.json)
  set(STB_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/stb")
  if(NOT EXISTS "${STB_DIR}/stb_image.h")
    include(FetchContent)
    FetchContent_Declare(stb
      GIT_REPOSITORY https://github.com/nothings/stb.git
      GIT_TAG master
      GIT_SHALLOW TRUE
      SOURCE_DIR "${STB_DIR}")
    FetchContent_MakeAvailable(stb)
  endif()

  add_executable(classify_image tools/classify_image.cpp)
  target_include_directories(classify_image PRIVATE "${STB_DIR}")
  target_link_libraries(classify_image PRIVATE crosschick_inference)
endif()
