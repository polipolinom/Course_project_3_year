cmake_minimum_required(VERSION 3.21.2)
project(course_project_3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_library(course_project_3_lib)

add_executable(course_project_3 src/main.cpp)
#target_link_libraries(course_project_3 course_project_3_lib)

add_executable(course_project_3_test tests/test.cpp tests/graphic_tests.cpp tests/random_objects.cpp)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(course_project_3_test gtest_main)
add_test(NAME test_all COMMAND course_project_3_test)

target_include_directories(course_project_3 PUBLIC SVD_course_project_2_year/course-project-second-year)
target_include_directories(course_project_3_test PUBLIC SVD_course_project_2_year/course-project-second-year)
add_subdirectory(SVD_course_project_2_year/course-project-second-year)
target_link_libraries(course_project_3 course_project_lib)
target_link_libraries(course_project_3_test course_project_lib)