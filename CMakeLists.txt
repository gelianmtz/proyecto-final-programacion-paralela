cmake_minimum_required(VERSION 3.16)
project(fractal_convolve_sequential LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fractal_convolve_sequential src/main.cpp)

if(MSVC)
  target_compile_options(fractal_convolve_sequential PRIVATE /O2 /W4)
else()
  target_compile_options(fractal_convolve_sequential PRIVATE -O2 -Wall -Wextra)
endif()
