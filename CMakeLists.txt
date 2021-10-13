cmake_minimum_required(VERSION 3.20)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX)

# You need this for using Kokkos_ROOT variable
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

add_executable(reduce_bool kk_reduce_bool.cpp)
add_executable(reduce_int kk_reduce_int.cpp)
add_executable(reduce_array kk_reduce_array.cpp)
add_executable(reduce_custom kk_custom_reduce.cpp)

# This is the only thing required to set up compiler/linker flags
target_link_libraries(reduce_bool Kokkos::kokkos)
target_link_libraries(reduce_array Kokkos::kokkos)
target_link_libraries(reduce_int Kokkos::kokkos)
target_link_libraries(reduce_custom Kokkos::kokkos)

enable_testing()
add_test(NAME redarray COMMAND reduce_array)
add_test(NAME redbool COMMAND reduce_bool)
add_test(NAME redint COMMAND reduce_int)
add_test(NAME redcustom COMMAND reduce_custom)
