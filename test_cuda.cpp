#include "test_cuda.h"
#pragma comment(lib, "cudart.lib")

int main(int argc, char**argv)
{
	std::string jpg_fn = argv[1];
	cv::Mat image = cv::imread(jpg_fn, CV_8U);
	int count = getImageNonzeroCount(image);
	std::cout << "count: " << count<< std::endl;
	return 0;
}