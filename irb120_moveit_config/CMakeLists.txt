cmake_minimum_required(VERSION 3.10.2)
project(irb120_moveit_config)

find_package(ament_cmake REQUIRED)

ament_package()

install(FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config rviz
  DESTINATION share/${PROJECT_NAME}
)
