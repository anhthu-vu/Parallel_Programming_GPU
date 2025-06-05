#include <stdio.h>
#include <math.h>
#include "LDLt.cu"

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


// Compute the design matrix
/*
In this code, we used double floating points (double) instead because we noticed that using float gives a numerical instability. 
*/
__global__ void design_matrix(double* d_matrix, double* S_training, double* I_training, double* t_training, int degree, int L){
  /*
  Parameters:
    - L: number of samples in the dataset
  */

  int dim = (degree+1)*(degree+2)*(degree+3)/6; // Number of columns of the design matrix
  int idx =  threadIdx.x + blockDim.x*blockIdx.x;
  double S, I, t;
  while (idx < L){
    S = S_training[idx];
    I = I_training[idx];
    t = t_training[idx];
    int d = 0;
    for (int i=0; i<degree+1; i++){
      for (int j=0; j<degree-i+1; j++){
        for (int k=0; k<degree-i-j+1; k++){
          double pow_S = 1.0f;
          double pow_I = 1.0f;
          double pow_t = 1.0f;
          for (int l=0; l<i; l++) {
            pow_S *= S;
          }
          for (int l=0; l<j; l++) {
            pow_I *= I;
          }
          for (int l=0; l<k; l++) {
            pow_t *= t;
          }
          d_matrix[idx*dim+d] = pow_S*pow_I*pow_t;
          d++;
        }
      }
    }
    idx += blockDim.x*gridDim.x;
  }
}


// Compute A^T@b where b can be y or A
__global__ void matmul(double* A, double* b, double* result_mat, int col_A, int col_b, int L){
  /*
  Parameters:
    - A is of size LxcolA
    - b is of size Lxcol_b

  The code is inspired from the code in `Add.cu` in which each thread is responsible for computing A_i[k]*b_j[k] 
  where A_i and b_j are the columns of A and b, respectively. The sum of those products is computed using reduction.
  */

  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  extern __shared__ double prod[];

  while (idx<L) { 
    for (int i=0; i<col_A; i++){
      for (int j=0; j<col_b; j++){
        prod[threadIdx.x] = A[idx*col_A+i] * b[idx*col_b+j];
        __syncthreads();
        
        int k = blockDim.x / 2;
        while (k != 0) {
          if (threadIdx.x<k) {
            prod[threadIdx.x] += prod[threadIdx.x+k];
          }
          __syncthreads();
          k /= 2;
        }

        if (threadIdx.x == 0) {
          atomicAdd(&result_mat[i*col_b+j], prod[0]); 
        }
        // Set all elements in the shared memory to 0 to avoid incorrect accumulations of inactive threads in the next loops
        prod[threadIdx.x] = 0.0f;  
        __syncthreads();
      }
    }
    idx += blockDim.x*gridDim.x; 
  }
}


int main(void) {
  /*
  The function LDLt_max_k(float *a, float *y, int n) in the `LDLt.cu` file returns
  the solutions of multiple linear systems using LDLt factorization
                    A_1*x = y_1
                    A_2*x = y_2
                    ...
                    A_k*x = y_k
    
    where the data for (A_1, ..., A_k) are stored in `a`, the data for
    (y_1, ..., y_k) are stored in `y`. Each A_i is a real symmetric matrix
    of dimension nxn, and each y_i is a vector of size nx1. 

    In the code of the function LDLt_max_k, each group of n threads 
    is responsible for processing a linear system A_i*x=y_i.
  
  Explanation of the indexing in `LDLt.cu`:
    + int Qt = threadIdx.x/d;
    + int gbx = Qt + blockIdx.x*(blockDim.x/d);
    => While the index `gbx` indicates the location where the data for
    the `gbx`-th system is stored in global memory (0 <= gbx <= (k-1)),
    Qt indicates the location where that data is stored in the share memory of a block.  
      
    + int tidx = threadIdx.x - Qt*d;
    If the upper triangular part of an A_i is as follows:
              (a_11 a_12 a_13 a_14
               0    a_22 a_23 a_24
               0    0    a_33 a_34
               0    0    0    a_44)
    => Then the index `tidx` indicates the diagonal 
    that the thread threadIdx.x is responsible for processing.
    (Note: the largest diagonal is indexed as 0).
  */


  // Read the test dataset
  int max_length = 500000;
  
  double* S_temp, *I_temp, *F_temp, *t_temp;
  
  cudaMallocManaged(&S_temp, max_length*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(S_temp, 0, max_length*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&I_temp, max_length*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(I_temp, 0, max_length*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&F_temp, max_length*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(F_temp, 0, max_length*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&t_temp, max_length*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(t_temp, 0, max_length*sizeof(double));
  testCUDA(cudaGetLastError());

  FILE *test_file = fopen("Test_dataset.csv", "r");

  if (test_file == NULL) {
    printf("Error: Could not open Test_dataset.csv\n");
    return EXIT_FAILURE;
  }

  int index = 0;
  char line[256];
  fgets(line, sizeof(line), test_file); // Skip header line

  while (fgets(line, sizeof(line), test_file)){
    if (sscanf(line, "%lf, %lf, %lf, %lf", &S_temp[index], &I_temp[index], &F_temp[index], &t_temp[index]) == 4){
      index++;
    }
  }
  fclose(test_file);

  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());
  
  printf ("Number of samples in the test dataset: %d\n", index);

  double* S_test, *I_test, *F_test, *t_test;
  
  cudaMallocManaged(&S_test, index*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(S_test, 0, index*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&I_test, index*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(I_test, 0, index*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&F_test, index*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(F_test, 0, index*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMallocManaged(&t_test, index*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(t_test, 0, index*sizeof(double));
  testCUDA(cudaGetLastError());

  cudaMemcpy(S_test, S_temp, index*sizeof(double), cudaMemcpyDeviceToDevice);
  testCUDA(cudaGetLastError());

  cudaMemcpy(I_test, I_temp, index*sizeof(double), cudaMemcpyDeviceToDevice);
  testCUDA(cudaGetLastError());

  cudaMemcpy(F_test, F_temp, index*sizeof(double), cudaMemcpyDeviceToDevice);
  testCUDA(cudaGetLastError());

  cudaMemcpy(t_test, t_temp, index*sizeof(double), cudaMemcpyDeviceToDevice);
  testCUDA(cudaGetLastError());

  cudaFree(S_temp);
  cudaFree(I_temp);
  cudaFree(F_temp);
  cudaFree(t_temp);
  

  // Compute the design matrix \phi(S_t, I_t, t)
  int NB = 1024;
  int NTPB = 512;
  int degree = 1;
  int dim = (degree+1)*(degree+2)*(degree+3)/6;

  double* d_matrix;
  cudaMallocManaged(&d_matrix, index*dim*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(d_matrix, 0, index*dim*sizeof(double));
  testCUDA(cudaGetLastError());

  design_matrix<<<NB, NTPB>>>(d_matrix, S_test, I_test, t_test, degree, index);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());


  // Compute d_matrix^t@d_matrix and d_matrix^t@F_test
  double* A;
  cudaMallocManaged(&A, dim*dim*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(A, 0, dim*dim*sizeof(double));
  testCUDA(cudaGetLastError());

  matmul<<<NB, NTPB, NTPB*sizeof(double)>>>(d_matrix, d_matrix, A, dim, dim, index);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());

  double* y;
  cudaMallocManaged(&y, dim*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(y, 0, dim*sizeof(double));
  testCUDA(cudaGetLastError());

  matmul<<<NB, NTPB, NTPB*sizeof(double)>>>(d_matrix, F_test, y, dim, 1, index); 
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());


  // Compute the solution
  NB = 1;
  NTPB = dim;
  LDLt_max_k<<<NB, NTPB, ((dim*dim+dim)/2+dim)*sizeof(double)>>>(A, y, dim);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError());

  cudaFree(A);

  // Compute the prediction 
  // There is `d_matrix_transpose` because the function matmul above compute A^T@b
  double* d_matrix_transpose;
  cudaMallocManaged(&d_matrix_transpose, index*dim*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(d_matrix_transpose, 0, index*dim*sizeof(double));
  testCUDA(cudaGetLastError());

  for (int i=0; i<index; i++){
    for (int j=0; j<dim; j++){
      d_matrix_transpose[j*index+i] = d_matrix[i*dim+j];
    }
  }

  cudaFree(d_matrix);

  double* predictions;
  cudaMallocManaged(&predictions, index*sizeof(double));
  testCUDA(cudaGetLastError());
  cudaMemset(predictions, 0, index*sizeof(double));
  testCUDA(cudaGetLastError());

  matmul<<<NB, NTPB, NTPB*sizeof(double)>>>(d_matrix_transpose, y, predictions, index, 1, dim);
  cudaDeviceSynchronize();
  testCUDA(cudaGetLastError()); 

  cudaFree(d_matrix_transpose);

  double error = 0.0f;

  for (int i=0; i<index; i++){
    double diff = predictions[i]-F_test[i];
    error += diff*diff/index;
  }

  printf ("The error on the test dataset is: %f\n", error);

  cudaFree(S_test);
  cudaFree(I_test);
  cudaFree(F_test);
  cudaFree(t_test);
  cudaFree(y);
  
  return 0;
}


