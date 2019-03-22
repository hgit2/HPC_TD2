#include "wb.h"

void vecAdd(float *in1, float *in2, float *out, int len) {
	vecAddKernel <<< ceil(len/256.0),256 >>> (in1 , in2 , out , len );
}

__global__ void vecAddKernel( float *in1, float *in2, float *out, int len){
	int i = threadIdx.x+blockDim.x∗blockIdx.x;
	if(i< len){
		out[i]=in1[i]+in2[i];
	}
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");

		// Rappel : CPU=host
    hostInput1 =
    (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
    (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);
    wbTime_start(GPU, "Allocating GPU memory.");

    //@@ Allocate GPU memory here
		cudaMalloc(( void ∗∗ ) &deviceInput1 , inputlength * sizeof(float));		
		cudaMalloc(( void ∗∗ ) &deviceInput2 , inputlength * sizeof(float));
		cudaMalloc(( void ∗∗ ) &deviceOutput , inputlength * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");
    wbTime_start(GPU, "Copying input memory to the GPU.");

    //@@ Copy memory to the GPU here
		cudaMemcpy( deviceInput1 , hostInput1 , inputlength * sizeof(float) , cudaMemcpyHostToDevice );
		cudaMemcpy( deviceInput2 , hostInput2 , inputlength  * sizeof(float), cudaMemcpyHostToDevice );

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
		// cf fonction vecadd

    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
		vecAdd(deviceInput1, deviceInput2, deviceOutput, inputlength)

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
		cudaMemcpy( hostOutput , deviceOutput , inputlength * sizeof(float) , cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");
    wbTime_start(GPU, "Freeing GPU Memory");

    //@@ Free the GPU memory here
		cudaFree( deviceInput1 ); 
		cudaFree( deviceInput2 );
		cudaFree( deviceOutput );

    wbTime_stop(GPU, "Freeing GPU Memory");
    wbSolution(args, hostOutput, inputLength);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
