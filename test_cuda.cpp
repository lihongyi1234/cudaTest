#include "test_cuda.h"
#pragma comment(lib, "cudart.lib")

int main(int argc, char**argv)
{
	std::string jpg_fn = argv[1];
	cv::Mat image = cv::imread(jpg_fn, CV_8U);
	int count = 0;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			uchar value = image.at<uchar>(row, col);
			if (value > 0) {
				count++;
			}

		}
	}
	std::cout << "count: " << count << std::endl;

	int new_count = getImageNonzeroCount(image);

	std::cout << "new_count: " << new_count<< std::endl;
	return 0;
}