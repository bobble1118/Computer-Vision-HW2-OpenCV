#include "func.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

void applyMeanFilter(const cv::Mat& input, cv::Mat& output, int kernel_size) {
    if (kernel_size <= 0) return;
    if (kernel_size % 2 == 0) kernel_size++; 
    int radius = kernel_size / 2;
    int kernel_area = kernel_size * kernel_size;

    output.create(input.size(), input.type());
    
    int rows = input.rows;
    int cols = input.cols;
    int channels = input.channels();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < channels; ++c) {
                long sum = 0;
                
                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int py = std::max(0, std::min(rows - 1, i + ky));
                        int px = std::max(0, std::min(cols - 1, j + kx));
                        
                        sum += input.data[(py * cols + px) * channels + c];
                    }
                }
                output.data[(i * cols + j) * channels + c] = static_cast<uchar>(sum / kernel_area);
            }
        }
    }
}

static void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    std::vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

static void mergeSortInternal(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortInternal(arr, l, m);
        mergeSortInternal(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// 2. Median Filter
void applyMedianFilter(const cv::Mat& input, cv::Mat& output, int kernel_size) {
    if (kernel_size <= 0) return;
    if (kernel_size % 2 == 0) kernel_size++; 
    
    int radius = kernel_size / 2;
    int total_pixels = kernel_size * kernel_size;
    
    output.create(input.size(), input.type());
    
    int rows = input.rows;
    int cols = input.cols;
    int channels = input.channels();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < channels; ++c) {
                std::vector<int> window;
                window.reserve(total_pixels);

                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {                        
                        int py = std::max(0, std::min(rows - 1, i + ky));
                        int px = std::max(0, std::min(cols - 1, j + kx));

                        int pixel_val = input.data[(py * cols + px) * channels + c];
                        window.push_back(pixel_val);
                    }
                }

                mergeSortInternal(window, 0, total_pixels - 1);
                output.data[(i * cols + j) * channels + c] = static_cast<uchar>(window[total_pixels / 2]);
            }
        }
    }
}

// 3. Draw Histogram
void drawHistogram(const cv::Mat& input, const std::string& filename) {
    // 1. 設定畫布尺寸與邊界
    const int canvas_w = 600;
    const int canvas_h = 600;
    const int margin = 75; // 四周預留 50 像素畫座標軸
    const int draw_w = canvas_w - 2 * margin;
    const int draw_h = canvas_h - 2 * margin;

    // 建立白色背景畫布
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Calculate histogram data
    std::vector<int> histData(256, 0);
    for (int r = 0; r < input.rows; ++r) {
        for (int c = 0; c < input.cols; ++c) {
            uchar val = input.at<uchar>(r, c);
            histData[val]++;
        }
    }

    // 2. 取得統計最大值以進行正規化 
    int max_val = *std::max_element(histData.begin(), histData.end());
    if (max_val == 0) max_val = 1; // 防止除以零

    // 3. 繪製座標軸 (L 型線條)
    // Y 軸
    cv::line(canvas, cv::Point(margin, margin), cv::Point(margin, canvas_h - margin), cv::Scalar(0, 0, 0), 2);
    // X 軸
    cv::line(canvas, cv::Point(margin, canvas_h - margin), cv::Point(canvas_w - margin, canvas_h - margin), cv::Scalar(0, 0, 0), 2);

    // 4. 繪製直方圖長條 
    double bin_w = (double)draw_w / 256;
    for (int i = 0; i < 256; i++) {
        int bar_h = (int)((double)histData[i] / max_val * draw_h);
        
        // 計算長條的左上角與右下角座標
        cv::Point pt1(margin + (int)(i * bin_w), canvas_h - margin - bar_h);
        cv::Point pt2(margin + (int)((i + 1) * bin_w), canvas_h - margin);

        // 使用藍色填充 
        cv::rectangle(canvas, pt1, pt2, cv::Scalar(200, 100, 0), -1);
    }

    // 5. 加入文字標籤與刻度
    // X 軸標籤
    cv::putText(canvas, "0", cv::Point(margin - 5, canvas_h - margin + 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "255", cv::Point(margin + draw_w - 15, canvas_h - margin + 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(canvas, "Pixel Value", cv::Point(canvas_w / 2 - 40, canvas_h - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // Y 軸標籤與輔助線 (每 20% 增加標籤與虛線)
    for (int i = 1; i <= 5; i++) {
        double ratio = i * 0.2;
        int val = (int)(max_val * ratio);
        int y = canvas_h - margin - (int)(ratio * draw_h);

        // 繪製水平虛線
        for (int x = margin; x < canvas_w - margin; x += 10) {
            cv::line(canvas, cv::Point(x, y), cv::Point(std::min(x + 5, canvas_w - margin), y), cv::Scalar(200, 200, 200), 1);
        }

        // Y 軸數值標籤
        cv::putText(canvas, std::to_string(val), cv::Point(5, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);
    }
    cv::putText(canvas, "Count", cv::Point(5, margin - 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    std::string outputPath = "project_hw2/result_img/" + filename + ".png";
    cv::imwrite(outputPath, canvas);
    std::cout << "Saved histogram plot: " << outputPath << std::endl;
}