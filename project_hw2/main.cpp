#include <iostream>
#include <opencv2/opencv.hpp>
#include "func.h"
#include <vector>

int main() {
    // Read Image
    cv::Mat noise_img = cv::imread("project_hw2/test_img/noise_image.png");

    if (noise_img.empty()) {
        noise_img = cv::imread("test_img/noise_image.png");
    }

    if (noise_img.empty()) {
        std::cerr << "Error: Could not find noise_image.png in test_img/" << std::endl;
        return -1;
    }

    std::cout << "--- Processing ---" << std::endl;

    // 1. Mean Filter
    cv::Mat output1;
    applyMeanFilter(noise_img, output1, 5);
    cv::imwrite("project_hw2/result_img/output1.png", output1);
    std::cout << "1. Mean Filter Done. Saved as result_img/output1.png" << std::endl;

    // 2. Median Filter
    cv::Mat output2;
    applyMedianFilter(noise_img, output2, 5);
    cv::imwrite("project_hw2/result_img/output2.png", output2);
    std::cout << "2. Median Filter Done. Saved as result_img/output2.png" << std::endl;

    // 3. Image Histograms Verification
    drawHistogram(noise_img, "noise_image_his");
    drawHistogram(output1, "output1_his");
    drawHistogram(output2, "output2_his");
    std::cout << "3. Histagrams Image Done." << std::endl;

    std::cout << "\n--- All Filter Tasks Finished ---" << std::endl;
    return 0;
}
