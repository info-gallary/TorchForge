# NeuroCUDA demo program.
# Multiply two random matrices, apply ReLU, and print the result.
# The optimizer fuses the matmul+relu pair below into a single
# `matmul_relu` CUDA kernel (see src/optimizer.py).
tensor A = random(1024, 1024)
tensor B = random(1024, 1024)
C = matmul(A, B)
D = relu(C)
print D
