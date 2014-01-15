#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define TILE 16 

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	
	__shared__ float SA[TILE][TILE];
	__shared__ float SB[TILE][TILE];
	
	int bx = blockIdx.x ; int by = blockIdx.y ;
	int tx = threadIdx.x; int ty = threadIdx.y; 
	
	int r =  by * TILE + ty;
	int c =  bx * TILE + tx;
	
	float pVal = 0.0; 

		for(int i = 0; i < (TILE + numAColumns-1)/TILE ; ++i)
		{
			if(r < numARows && i*TILE+ tx < numAColumns)				
				SA[ty][tx] = A[r * numAColumns +i*TILE+ tx];
			else SA[ty][tx] = 0.0;
			
			if((i*TILE + ty) < numBRows && c < numBColumns)
				SB[ty][tx] = B[(i*TILE + ty)*numBColumns + c];
			else SB[ty][tx] = 0.0;
			
			__syncthreads();
		
			for(int k = 0; k < TILE; ++k)
			{
				pVal += SA[ty][k]*SB[k][tx];
			}
			__syncthreads();
		}
		
	  if (r < numCRows && c < numCColumns)
		  C[((by * blockDim.y + threadIdx.y)*numCColumns)+(bx*blockDim.x)+tx] = pVal;
	
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
	hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	int sizeA = numAColumns * numARows * sizeof(float);
    int sizeB = numBColumns * numBRows * sizeof(float);
	int sizeC = numCColumns * numCRows * sizeof(float);
	cudaMalloc((void**)&deviceA,sizeA);
	cudaMalloc((void**)&deviceB,sizeB);
	cudaMalloc((void**)&deviceC,sizeC);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA,sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB,sizeB, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceC, hostC,sizeC, cudaMemcpyHostToDevice);
	
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
     dim3 dimBlock(16,16,1);
	 dim3 dimGrid((numCColumns+dimBlock.x-1)/dimBlock.x,(numCRows+ dimBlock.y -1)/dimBlock.y,1);	

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiply<<<dimGrid,dimBlock>>>(deviceA,deviceB, deviceC,numARows, numAColumns,numBRows, numBColumns,numCRows,numCColumns);
   
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
	
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}