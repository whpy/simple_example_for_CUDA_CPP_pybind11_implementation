#include<cuda_runtime.h>
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}

// __global__ void add2_kernel(float* c,
//                             const float* a,
//                             const float* b,
//                             int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i<n) {
//         c[i] = a[i] + b[i];
//     }
// }

// void launch_add2(float* c,
//                  const float* a,
//                  const float* b,
//                  int n) {


//     dim3 grid((n + 1023) / 1024);
//     dim3 block(1024);
//     int N = n;

//     float *d_a, *d_b, *d_c;
//     cudaMalloc((void**)&d_a, sizeof(float)*N);
//     cudaMalloc((void**)&d_b, sizeof(float)*N);
//     cudaMalloc((void**)&d_c, sizeof(float)*N);

//     cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);
//     add2_kernel<<<1, N>>>(d_c, d_a, d_b, n);
//     // cudaDeviceSynchronize();
//     cudaMemcpy(c, d_c, sizeof(float)*N, cudaMemcpyDeviceToHost);
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
// }