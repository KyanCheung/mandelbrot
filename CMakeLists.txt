cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(mandelbrot LANGUAGES CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

set (CMAKE_CXX_STANDARD 20)

add_executable(mandelbrot mandelbrot.cu)
add_executable(mandelbrot-double mandelbrot-double.cu)
add_executable(multibrot multibrot.cu)
add_executable(negabrot negabrot.cu)

find_package(PNG)
include_directories(${PNG_INCLUDE_DIRS})
target_link_libraries(mandelbrot ${PNG_LIBRARIES})
target_link_libraries(mandelbrot-double ${PNG_LIBRARIES})
target_link_libraries(multibrot ${PNG_LIBRARIES})
target_link_libraries(negabrot ${PNG_LIBRARIES})