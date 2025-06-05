#include <stdio.h>
#include <curand_kernel.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
    printf("Error in file %s at line %d\n", file, line);
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


// Set the state for each thread
__global__ void init_curand_nested_state_k(int seed, curandState* state){
	int idx = blockDim.x*blockIdx.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}


// Monte Carlo simulation kernel 
__global__ void MC_k2(float S_0, float sigma, float dt, int N, curandState* state,
float* sum_F, float* S_t, float* I_t, float* t, int nb_loops, int nb_traj_per_loop){
  /*
  Idea: 
    + The blocks in the same column simulate samples for the same timestep t
    + Each thread simulate a trajectory
    Thus, this process collects gridDim.y*blockDim.x samples for each timestep t. It will be repeated `nb_loops` times to collect more samples.
  Parameters:
    + nb_traj_per_loop = gridDim.y*blockDim.x*nb_timesteps
  */

	int idx = blockDim.x*blockIdx.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;
  curandState localState = state[idx];
  float2 G;
  float S, I;
  int gb_idx; 

  for (int j=0; j<nb_loops; j++){
    S = S_0;
    I = 0.0f;
    // Index for each sample (S_t, I_t, t) in global memory
    gb_idx = j*nb_traj_per_loop + idx;
    for (int i=0; i<N; i++){
      G = curand_normal2(&localState);
      I = (float)i/(i+1)*I + (float)1/(i+1)*S;
      S *= (1 + sigma*dt*G.x);
      if (blockIdx.x == i){
        I_t[gb_idx] = I;
        S_t[gb_idx] = S;
        t[gb_idx] = (float)(i+1)/N;
      }
    } 
    sum_F[gb_idx] = fmaxf(0.0f, S-I);
  }
}


__global__ void MC_nested_k(float S_0, float sigma, float dt, int N, curandState* state,
float* sum_F, float* S_t, float* I_t, float* t, int nb_blocks, int nb_timesteps, int nb_loops_nested, int nb_loops_timestep, int n){
  /*
  Idea: For each timestep t,
    + The thread indexed 0 of each block simulates a sample (S_t, I_t), then save the results in the shared memeory.
    + After that, in each block, each thread simulates a inner MC trajectory starting from (S_t, I_t) that Thread 0 have collected,
    then performs the reduction on memory to compute E[(S_T-I_T)_+ | S_t, I_t]. This process will be repeated for `nb_loops_nested` times for more precision. 
    Thus, this process collects nb_blocks = gridDim.x*gridDim.y samples (S_t, I_t). It will be repeated `nb_loops_timestep` times to collect more samples.

  Paramters: 
    - nb_blocks: the number of blocks in a grid
    - nb_timsteps: the number of timesteps (\delta t, 2*\delta t, ..., 99*\delta t)
    - nb_loops_nested: the number of loops for the inner MC simulations  
    - nb_loops_timestep: the number of loops for the outer MC simulations
    - n = nb_loops_nested*blockDim.x
  */
	int idx = blockDim.x*blockIdx.x*gridDim.y + blockDim.x*blockIdx.y + threadIdx.x;
  curandState localState = state[idx];
  float2 G;
  float S, I;
  int gb_block_idx;
  extern __shared__ float A[];

  float* R1s, *S_I_temp;
  R1s = A;
  S_I_temp = R1s + blockDim.x;

  for (int step=1; step<nb_timesteps+1; step++){
    for (int k=0; k<nb_loops_timestep; k++){
      S = S_0;
      I = 0.0f;
      // Index for each sample (S_t, I_t, t) in global memory
      gb_block_idx = blockIdx.y + blockIdx.x*gridDim.y + k*nb_blocks + (step-1)*nb_loops_timestep*nb_blocks;
      if (threadIdx.x == 0){
        for (int s=0; s<step; s++){
          G = curand_normal2(&localState);
          I = (float)s/(s+1)*I + (float)1/(s+1)*S;
          S *= (1 + sigma*dt*G.x);
        }
        I_t[gb_block_idx] = I;
        S_t[gb_block_idx] = S;
        t[gb_block_idx] = (float)step/N;
        S_I_temp[0] = S;
        S_I_temp[1] = I;
      }
      __syncthreads();
      
      for (int j=0; j<nb_loops_nested; j++){
        S = S_I_temp[0];
        I = S_I_temp[1];
        for (int i=step; i<N; i++){
          G = curand_normal2(&localState);
          I = (float)i/(i+1)*I + (float)1/(i+1)*S;
          S *= (1 + sigma*dt*G.x);
        }
        R1s[threadIdx.x] = fmaxf(0.0f, S-I)/n;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i!=0){
          if(threadIdx.x < i){
            R1s[threadIdx.x] += R1s[threadIdx.x+i];
          }
          __syncthreads();
          i /= 2;
        }
        if (threadIdx.x == 0) {
          atomicAdd(sum_F+gb_block_idx, R1s[0]);
        }
      } 
    }  
  }
}
  

int main(void) {
	float T = 1.0f;
	float S_0 = 100.0f;
	float sigma = 0.2f;
	int N = 100;
	float dt = sqrtf(T/N);
	
  // Simulate the training dataset
  int NTPB = 512;
  int nb_timesteps = 100;
  int nb_rows = 8;
  int nb_loops = 8;
  int nb_traj_per_loop = nb_rows*nb_timesteps*NTPB;
  dim3 Nblocks(nb_timesteps, nb_rows);

  float* S_t, *I_t, *sum_F, *t;
  cudaMallocManaged(&sum_F, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(sum_F, 0, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&S_t, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(S_t, 0, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&I_t, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(I_t, 0, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&t, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(t, 0, nb_traj_per_loop*nb_loops*sizeof(float));
  testCUDA(cudaGetLastError());

  curandState* states;
  cudaMalloc(&states, nb_traj_per_loop*sizeof(curandState));
  testCUDA(cudaGetLastError());
  init_curand_nested_state_k<<<Nblocks, NTPB>>>(0, states);
  testCUDA(cudaGetLastError());

  MC_k2<<<Nblocks, NTPB>>>(S_0, sigma, dt, N, states, sum_F, S_t, I_t, t, nb_loops, nb_traj_per_loop);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());

  FILE* training_file;
	training_file = fopen("Training_dataset.csv", "w+");
	fprintf(training_file, "S_t, I_t, F, t\n");
  for (int k=0; k<nb_traj_per_loop*nb_loops; k++) {
    fprintf(training_file, "%f, %f, %f, %f\n", S_t[k], I_t[k], sum_F[k], t[k]);
  }
  fclose(training_file);

  cudaFree(states);
  cudaFree(sum_F);
  cudaFree(S_t);
  cudaFree(I_t);
  cudaFree(t);


  // Simulate the validation dataset
  NTPB = 512;
  nb_timesteps = 100;
  nb_rows = 8;
  int nb_cols = 64;
  int nb_loops_nested = 32;
  int nb_loops_timestep = 8;
  int n = nb_loops_nested*NTPB;
  dim3 NBlocks(nb_cols, nb_rows);

  cudaMallocManaged(&sum_F, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(sum_F, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&S_t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(S_t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&I_t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(I_t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMalloc(&states, nb_rows*nb_cols*NTPB*sizeof(curandState));
  testCUDA(cudaGetLastError());
  init_curand_nested_state_k<<<NBlocks, NTPB>>>(1, states);
  testCUDA(cudaGetLastError());

  MC_nested_k<<<NBlocks, NTPB, (NTPB+2)*sizeof(float)>>>(S_0, sigma, dt, N, states, sum_F, S_t, I_t, t, nb_rows*nb_cols, nb_timesteps, nb_loops_nested, nb_loops_timestep, n);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());

  FILE* validation_file;
	validation_file = fopen("Validation_dataset.csv", "w+");
	fprintf(validation_file, "S_t, I_t, F, t\n");
  for (int k=0; k<nb_timesteps*nb_loops_timestep*nb_rows*nb_cols; k++) {
    fprintf(validation_file, "%f, %f, %f, %f\n", S_t[k], I_t[k], sum_F[k], t[k]);
  }
  fclose(validation_file);

  cudaFree(states);
  cudaFree(sum_F);
  cudaFree(S_t);
  cudaFree(I_t);
  cudaFree(t);


  // Simulate the test dataset
  cudaMallocManaged(&sum_F, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(sum_F, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&S_t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(S_t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&I_t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(I_t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&t, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());
  cudaMemset(t, 0, nb_timesteps*nb_loops_timestep*nb_rows*nb_cols*sizeof(float));
  testCUDA(cudaGetLastError());

  cudaMalloc(&states, nb_rows*nb_cols*NTPB*sizeof(curandState));
  testCUDA(cudaGetLastError());
  init_curand_nested_state_k<<<NBlocks, NTPB>>>(2, states);
  testCUDA(cudaGetLastError());

  MC_nested_k<<<NBlocks, NTPB, (NTPB+2)*sizeof(float)>>>(S_0, sigma, dt, N, states, sum_F, S_t, I_t, t, nb_rows*nb_cols, nb_timesteps, nb_loops_nested, nb_loops_timestep, n);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());

  FILE* test_file;
	test_file = fopen("Test_dataset.csv", "w+");
	fprintf(test_file, "S_t, I_t, F, t\n");
  for (int k=0; k<nb_timesteps*nb_loops_timestep*nb_rows*nb_cols; k++) {
    fprintf(test_file, "%f, %f, %f, %f\n", S_t[k], I_t[k], sum_F[k], t[k]);
  }
  fclose(test_file);

  cudaFree(states);
  cudaFree(sum_F);
  cudaFree(S_t);
  cudaFree(I_t);
  cudaFree(t);

	return 0;
}