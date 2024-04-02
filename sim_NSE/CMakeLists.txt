set(TARGETS
        sim_1
        sim_2
        sim_3
        sim_4
        sim_5
)

if(NOT HA_CAPACITY_VARIANT)
set (HA_CAPACITY_VARIANT 1)
endif()

if(NOT HA_VARIANT)
set (HA_VARIANT 1)
endif()


foreach(target IN ITEMS ${TARGETS})
    add_executable(${target} ${target}.cu)
    target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
    target_compile_definitions(${target} PUBLIC -DHA_CAPACITY_VARIANT=${HA_CAPACITY_VARIANT} -DHA_VARIANT=${HA_CAPACITY_VARIANT})
endforeach()
