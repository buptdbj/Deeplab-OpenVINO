#pragma once
#include <iostream>
#include <string>

using namespace std;

namespace tools{

/*****************************************
******************************************
 * @brief Writes output data to image
 * @param name - image name
 * @param data - output data
 * @param classesNum - the number of classes
 * @return false if error else true
******************************************
******************************************/
void overlayOutput(cv::Mat& data, cv::Mat& src, size_t classesNum) {
    unsigned int seed = (unsigned int) time(NULL);
    // Known colors for training classes from Cityscape dataset
    static std::vector<Color> colors = {
        {128, 64,  128},
        {232, 35,  244},
        {70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        {30,  170, 250},
        {0,   220, 220},
        {35,  142, 107},
        {152, 251, 152},
        {180, 130, 70},
        {60,  20,  220},
        {0,   0,   255},
        {142, 0,   0},
        {70,  0,   0},
        {100, 60,  0},
        {90,  0,   0},
        {230, 0,   0},
        {32,  11,  119},
        {0,   74,  111},
        {81,  0,   81}
    };

    while (classesNum > colors.size()) {
        static std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        Color color(dist(rng), dist(rng), dist(rng));
        colors.push_back(color);
    }

    int height = data.rows;
    int width = data.cols;

    cv::Mat3b Seg(height, width);
    cv::Mat3b Dst(height, width);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            cv::Vec3b color;
            size_t index = data.at<u_int8_t>(y, x);
            color[0] = colors.at(index).red();
            color[1] = colors.at(index).green();
            color[2] = colors.at(index).blue();
            Seg.at<cv::Vec3b>(cv::Point(x, y)) = color;
        }
    }

    double alpha = 0.6;
    double beta = 1 - alpha;
    cv::addWeighted(Seg, alpha, src, beta, 0.0, Dst);

    cv::imshow("Seg", Seg);
    cv::imshow("Overlay", Dst);
    cv::waitKey(0);
}

}