set(OP_NAME "block_sparse_attention")
set(OP_HAS_WORKSPACE TRUE)
set(OP_HOST_SRCS "${CMAKE_CURRENT_LIST_DIR}/op_host/block_sparse_attention.cpp")
set(OP_KERNEL_SRCS "${CMAKE_CURRENT_LIST_DIR}/op_kernel/block_sparse_attention.cpp")
