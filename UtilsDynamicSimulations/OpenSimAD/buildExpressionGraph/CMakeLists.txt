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

set(SDK_DIR "Path to OpenSim include directory" CACHE PATH "Location of directory")

target_compile_definitions(${TARGET_NAME} PUBLIC SimTK_REAL_IS_ADOUBLE)

if(WIN32) 
	target_include_directories(${TARGET_NAME} PUBLIC ${SDK_DIR}/include ${SDK_DIR}/Simbody/include)
	target_link_libraries(${TARGET_NAME} PUBLIC ${SDK_DIR}/lib/osimCommon_recorder.lib ${SDK_DIR}/lib/osimSimulation_recorder.lib ${SDK_DIR}/Simbody/lib/SimTKcommon_recorder.lib ${SDK_DIR}/Simbody/lib/SimTKmath_recorder.lib ${SDK_DIR}/Simbody/lib/SimTKsimbody_recorder.lib)
else()
	target_include_directories(${TARGET_NAME} PUBLIC ${SDK_DIR}/include ${SDK_DIR}/include/simbody)
	if(APPLE)
		target_link_libraries(${TARGET_NAME} PUBLIC ${SDK_DIR}/lib/libosimCommon_recorder.dylib ${SDK_DIR}/lib/libosimSimulation_recorder.dylib ${SDK_DIR}/lib/libSimTKcommon_recorder.dylib ${SDK_DIR}/lib/libSimTKmath_recorder.dylib ${SDK_DIR}/lib/libSimTKsimbody_recorder.dylib)
	elseif(UNIX)
		target_link_libraries(${TARGET_NAME} PUBLIC ${SDK_DIR}/lib/libosimCommon_recorder.so ${SDK_DIR}/lib/libosimSimulation_recorder.so ${SDK_DIR}/lib/libSimTKcommon_recorder.so ${SDK_DIR}/lib/libSimTKmath_recorder.so ${SDK_DIR}/lib/libSimTKsimbody_recorder.so)
	endif()
	
	add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${TARGET_NAME}
        )
endif()
		