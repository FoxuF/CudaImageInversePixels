
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>

void* buffers[3];

__global__ void inverseImage(uchar* src_img, int col, int row) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < col && j < row)
	{
		src_img[(col * j) + i] = 255 - src_img[(col * j) + i];
	}

}

int main() {
	cv::Mat imagen, imagenGris;

	imagen = cv::imread("C:/Users/ianli/Downloads/rookie.jpg");

	cv::cvtColor(imagen, imagenGris, cv::COLOR_BGR2GRAY);

	int memory = imagenGris.rows * imagenGris.cols;

	buffers[0] = malloc(memory);
	cudaMalloc(&buffers[1], memory);
	cudaMalloc(&buffers[2], memory);

	cudaMemcpy(buffers[1], imagenGris.ptr(), memory, cudaMemcpyHostToDevice);

	int thread = 32;
	dim3 threads(thread, thread);
	dim3 blocks((imagenGris.cols + thread - 1) / thread, (imagenGris.rows + thread -1)/thread);

	inverseImage << <blocks, threads >> > ((uchar*)buffers[1], imagenGris.cols, imagenGris.rows);

	cudaMemcpy(buffers[0], buffers[1], memory, cudaMemcpyDeviceToHost);

	cv::Mat salida = cv::Mat(cv::Size(imagen.cols, imagen.rows), CV_8U);

	std::memcpy(salida.data, buffers[0], memory);

	cv::imshow("ventanita OG", imagen);
	cv::imshow("ventanita", salida);
	cv::waitKey(0);
	return 0;

}