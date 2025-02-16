#ifndef QR_DETECTOR_H_
#define QR_DETECTOR_H_

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

namespace {
    constexpr size_t MAX_NUM_QR_CODES = 9;
    const std::array<cv::Scalar, MAX_NUM_QR_CODES> box_colors = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(128, 0, 0),
        cv::Scalar(0, 128, 0),
        cv::Scalar(0, 0, 128),
    };
}

class QRDetector {
public:
    QRDetector();
    ~QRDetector();

    void image_cb(const sensor_msgs::ImageConstPtr& msg, ros::Publisher& qr_pub, image_transport::Publisher& image_pub);

    struct Detection {
        std::string data;
        cv::Rect box;

        Detection() {}
        Detection(const std::string& data, cv::Rect box) : data(data), box(box) {}
    };

protected:
    void detect(cv::Mat &image, std::vector<Detection> &output);
    cv::QRCodeDetector qr_detector_;
};

#endif  // QR_DETECTOR_H_