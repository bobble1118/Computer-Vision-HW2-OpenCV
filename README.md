# CV_HW2: Image Filtering and Histograms

This project implements primitive image processing filtering algorithms in C++ from scratch, specifically without using OpenCV's built-in processing functions (e.g., `cv::blur` or `cv::medianBlur`). OpenCV is only used for basic I/O (reading and writing images), drawing, and matrix data structures.

## Overview

The following core computer vision tasks are implemented manually in C++:

1. **Mean Filter (`applyMeanFilter`)**: Applies a configurable Mean Filter (average filter) given a `kernel_size` and `stride` to smooth the image.
2. **Median Filter (`applyMedianFilter`)**: Applies a Median Filter to reduce noise (like Salt-and-Pepper noise) given a `kernel_size` and `stride`. This function also utilizes a completely custom implementation of the **Merge Sort** algorithm (`mergeSortInternal`) to find the median value within each kernel window, avoiding standard library sort functions.
3. **Draw Histogram (`drawHistogram`)**: Calculates the pixel intensity distribution of an image from scratch and manually plots a histogram chart using OpenCV's basic drawing functions (lines, rectangles, text). It includes dynamic axes, horizontal grid dashed lines at 20% intervals, and value markers.

## Folder Structure

- `project_hw2/`
  - `main.cpp`: Main executable file that reads testing images and passes them through the filtering pipeline.
  - `func.cpp`: Contains the manual implementation of all algorithm logic (filters, sorting, and histogram drawing).
  - `include/func.h`: Function declarations.
  - `test_img/`: Directory containing the input images (e.g., `noise_image.png`).
- `result_img/`: Output destination where all processed images and generated histograms will be saved.
- `CMakeLists.txt`: CMake build instructions.

## Prerequisites

- **CMake**: version >= 3.10
- **C++ Compiler**: C++11 compliant
- **OpenCV**: Built correctly and discoverable by CMake (Make sure OpenCV paths are correctly setup in your environment or `CMakeLists.txt`).

## Build Instructions

Using CMake from the terminal or Command Prompt, run the following script in the root repository directory:

```bash
# 1. Create a build directory
mkdir build
cd build

# 2. Generate the build files
cmake ..

# 3. Build the project
cmake --build . --config Release
```

## Running the Application

After building successfully, run the generated executable:

```bash
.\build\Release\cv_hw2.exe
```

The application will read the noise image from the `project_hw2/test_img` or `test_img` directory. It will then output the processed results and dynamically generated histogram charts into the `result_img` directory.

### Processed Output Sequence:
- `output1.png`: Image after Mean Filter
- `output2.png`: Image after Median Filter
- `noise_image_his.png`: Histogram plot of the original noisy image
- `output1_his.png`: Histogram plot of the Mean Filtered output
- `output2_his.png`: Histogram plot of the Median Filtered output
