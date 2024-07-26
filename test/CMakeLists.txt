set(UNIT_TEST_NAME UnitTest)

find_package(GTest REQUIRED)

if(NOT GTest_FOUND)
	message(FATAL_ERROR "Failed to find GTest...")
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)

enable_testing()

add_executable(
	${UNIT_TEST_NAME}
	${PROJECT_SOURCE_DIR}/test/KdTreeFLANNTests.cc
	${PROJECT_SOURCE_DIR}/src/KdTreeFLANN.cpp
)

target_link_libraries(${UNIT_TEST_NAME} flann ${LZ4_LIBRARIES} GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${UNIT_TEST_NAME})
