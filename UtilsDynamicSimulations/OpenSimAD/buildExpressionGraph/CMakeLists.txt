cmake_minimum_required (VERSION 3.10)
set (CMAKE_CXX_STANDARD 11)

set(TARGET_NAME 
    "" 
    CACHE 
    STRING
    "Target name")
    
set(TEST_TARGET ${TARGET_NAME})
project(${TEST_TARGET})

set(CPP_DIR "Path to external function directory" CACHE PATH "Location of directory")
add_executable(${TARGET_NAME} ${CPP_DIR}/${TARGET_NAME}.cpp)
# add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)

set(SDK_DIR "Path to OpenSim include directory" CACHE PATH "Location of directory")
target_include_directories(${TARGET_NAME} PUBLIC ${SDK_DIR}/include ${SDK_DIR}/Simbody/include)
target_compile_definitions(${TARGET_NAME} PUBLIC SimTK_REAL_IS_ADOUBLE)

target_link_libraries(${TARGET_NAME} PUBLIC ${SDK_DIR}/lib/osimCommon.lib ${SDK_DIR}/lib/osimSimulation.lib ${SDK_DIR}/Simbody/lib/SimTKcommon_recorder.lib ${SDK_DIR}/Simbody/lib/SimTKmath_recorder.lib ${SDK_DIR}/Simbody/lib/SimTKsimbody_recorder.lib)
		