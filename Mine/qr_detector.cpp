#include "qr_detector.h"

QRDetector::QRDetector() {
    ROS_INFO("QRDetector initialized");
}

QRDetector::~QRDetector() {
    ROS_INFO("QRDetector destroyed");
}

void QRDetector::detect(cv::Mat& image, std::vector<Detection>& output) {
    std::vector<cv::Point> points;
    std::string data = qr_detector_.detectAndDecode(image, points);
    if (!data.empty()) {
        cv::Rect box = cv::boundingRect(points);
        output.emplace_back(data, box);
    }
}

void QRDetector::image_cb(const sensor_msgs::ImageConstPtr& msg, ros::Publisher& qr_pub, image_transport::Publisher& image_pub) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat frame = cv_ptr->image;
    std::vector<Detection> output;
    detect(frame, output);

    for (const auto& detection : output) {
        auto box = detection.box;
        const auto color = box_colors[0]; // Assuming single QR code for simplicity
        cv::rectangle(frame, box, color, 3);
        cv::putText(frame, detection.data, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }

    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    image_pub.publish(img_msg);
}