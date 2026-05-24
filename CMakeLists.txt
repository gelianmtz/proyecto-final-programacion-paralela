cmake_minimum_required(VERSION 3.16)
project(fractal_convolve_parallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(fractal_convolve_parallel src/main.cpp)
target_link_libraries(fractal_convolve_parallel PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(fractal_convolve_parallel PRIVATE /O2 /W4)
else()
  target_compile_options(fractal_convolve_parallel PRIVATE -O2 -Wall -Wextra)
endif()
