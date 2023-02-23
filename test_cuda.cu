#pragma once
#include "test_cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>

__inline__ __device__ int warpReduceSum(int val)
{
	int warpSize = 32;
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

/** \brief Returns the warp lane ID of the calling thread. */
static __device__ __forceinline__ unsigned int
laneId()
{
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
			}

static __device__ __forceinline__
int laneMaskLt()
{
#if (__CUDA_ARCH__ >= 200)
	unsigned int ret;
	asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
	return ret;
#else
	return 0xFFFFFFFF >> (32 - laneId());
#endif
}

static __device__ __forceinline__ int binaryExclScan(int ballot_mask)
{
	return __popc(laneMaskLt() & ballot_mask);
}

__global__ void getImageNonzeroCountKernel(cudaTextureObject_t texObj, int* points, int* indices, int* nonzero_count, const int width,const int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	int idx = y * width + x;
	int lane = threadIdx.x;
	int wid = threadIdx.y;
	volatile __shared__ int warps_buffer[32];
	int tag = tex2D<int>(texObj, x + 0.5f, y + 0.5f);
	bool valid = (tag == 0) ? false : true;
	int val = (tag == 0) ? 0 : 1;
	int sum = warpReduceSum(val);
	if (sum == 0) { return; }
	if (lane == 0) {
		int old = atomicAdd(nonzero_count, sum);
		warps_buffer[wid] = old;
	}
	int old_global_count = warps_buffer[wid];
	int offs = binaryExclScan(__ballot_sync(0xFFFFFFFF, valid));
	if (old_global_count + offs < height * width && valid) {
		int out_idx = old_global_count + offs;
		points[out_idx] = tag;
		indices[out_idx] = idx;
	}
}


void uploadMat(const cv::Mat& image, cudaArray*& cuArray, cudaTextureObject_t& texture)
{
	//The texture description
	cudaTextureDesc uchar1_texture_desc;
	memset(&uchar1_texture_desc, 0, sizeof(uchar1_texture_desc));
	uchar1_texture_desc.addressMode[0] = cudaAddressModeClamp;
	uchar1_texture_desc.addressMode[1] = cudaAddressModeClamp;
	uchar1_texture_desc.addressMode[2] = cudaAddressModeClamp;
	uchar1_texture_desc.filterMode = cudaFilterModePoint;
	uchar1_texture_desc.readMode = cudaReadModeElementType;
	uchar1_texture_desc.normalizedCoords = 0;
	
	//Create channel descriptions
	cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	//Create the resource desc
	cudaResourceDesc resource_desc;
	//cudaArray* _cuArray;
	cudaMallocArray(&cuArray, &uchar1_channel_desc, image.cols, image.rows);
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = cuArray;

	//Allocate the texture
	//cudaTextureObject_t _texture = 0;
	cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0);
	cudaMemcpyToArray(cuArray, 0, 0, image.data, sizeof(uchar) * image.rows * image.cols, cudaMemcpyHostToDevice);

}

int getImageNonzeroCount(cv::Mat& image)
{
	int width = image.cols;
	int height = image.rows;
	std::cout << "width:" << width << " height:" << height << std::endl;
	
	//host
	int init_count = width * height;
	int* _points = new int[init_count];
	int* _indices = new int[init_count];
	int nonzero_count = 0;

	//device
	int* d_nonzero_count;
	int* d_points, * d_indices;
	cudaMalloc((void**)&d_nonzero_count, sizeof(int));
	cudaMalloc((void**)&d_points, init_count * sizeof(int));
	cudaMalloc((void**)&d_indices, init_count * sizeof(int));

	//upload cv::Mat uchar to device 
	cudaArray* cuArray;
	cudaTextureObject_t texture;
	uploadMat(image, cuArray, texture);

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	getImageNonzeroCountKernel << <numBlocks, threadsPerBlock >> > (texture, d_points, d_indices, d_nonzero_count, width, height);
	cudaStreamSynchronize(0);

	//copy device data to host
	cudaMemcpy(&nonzero_count, d_nonzero_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)_points, (void*)d_points, nonzero_count * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)_indices, (void*)d_indices, nonzero_count * sizeof(int), cudaMemcpyDeviceToHost);

	
	cudaDestroyTextureObject(texture);
	cudaFreeArray(cuArray);

	delete[] _points;
	delete[] _indices;
	cudaFree(d_nonzero_count);
	cudaFree(d_points);
	cudaFree(d_indices);
	return nonzero_count;
}
