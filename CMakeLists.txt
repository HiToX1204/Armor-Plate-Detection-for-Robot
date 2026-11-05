cmake_minimum_required(VERSION 3.16)
project(hongxihe_marker_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(marker_recognition main.cpp)
target_include_directories(marker_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(marker_recognition PRIVATE ${OpenCV_LIBS})

if (MSVC)
    target_compile_options(marker_recognition PRIVATE /W4)
else()
    target_compile_options(marker_recognition PRIVATE -Wall -Wextra -Wpedantic)
endif()
