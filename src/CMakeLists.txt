add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
#    visual_odometry_PnP.cpp
    visual_odometry_ICP.cpp
#    visual_odometry_ICP_BA_revised.cpp
)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)
