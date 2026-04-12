#ifndef FUNC_H
#define FUNC_H
#include <opencv2/opencv.hpp>
#include <vector>

void applyMeanFilter(const cv::Mat& input, cv::Mat& output, int kernel_size = 3, int stride = 1);
void applyMedianFilter(const cv::Mat& input, cv::Mat& output, int kernel_size = 3, int stride = 1);
void drawHistogram(const cv::Mat& input, const std::string& filename);

#endif