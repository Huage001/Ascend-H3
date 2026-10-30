set(OP_NAME "spark_joint_v4_grad")
set(OP_HAS_WORKSPACE TRUE)
set(OP_HOST_SRCS "${CMAKE_CURRENT_LIST_DIR}/op_host/spark_joint_v4_grad.cpp")
set(OP_KERNEL_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/op_kernel/spark_joint_v4_grad.cpp")
