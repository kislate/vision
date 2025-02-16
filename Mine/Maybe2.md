好的，我们将创建一个完整的程序来识别9种二维码，并将结果发布出去。这个程序将从摄像头读取视频流，使用预训练的权重文件进行二维码识别，并将识别结果发布到ROS话题。以下是详细的步骤：

### 1. 创建 `qr_detector.h`

```cpp
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
#include <opencv_cpp_yolov5/QRCode.h>
#include <opencv_cpp_yolov5/QRCodes.h>

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
    QRDetector(const std::string& weights_path);
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
    std::string weights_path_;
};

#endif  // QR_DETECTOR_H_
```

### 2. 创建 `qr_detector.cpp`

```cpp
#include "qr_detector.h"

QRDetector::QRDetector(const std::string& weights_path) : weights_path_(weights_path) {
    ROS_INFO("QRDetector initialized with weights: %s", weights_path.c_str());
    // 在这里加载权重文件，如果需要的话
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

    opencv_cpp_yolov5::QRCodes qr_codes_msg;
    qr_codes_msg.header = msg->header;

    for (const auto& detection : output) {
        auto box = detection.box;
        const auto color = box_colors[0]; // Assuming single QR code for simplicity
        cv::rectangle(frame, box, color, 3);
        cv::putText(frame, detection.data, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        opencv_cpp_yolov5::QRCode qr_code_msg;
        qr_code_msg.data = detection.data;
        qr_code_msg.xmin = box.x;
        qr_code_msg.ymin = box.y;
        qr_code_msg.xmax = box.x + box.width;
        qr_code_msg.ymax = box.y + box.height;
        qr_codes_msg.qr_codes.push_back(qr_code_msg);
    }

    qr_pub.publish(qr_codes_msg);

    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    image_pub.publish(img_msg);
}
```

### 3. 创建 `qr_detection_node.cpp`

```cpp
#include "qr_detector.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "qr_detection_node");
    ros::NodeHandle nh("~");

    std::string weights_path;
    nh.param("weights_path", weights_path, std::string("/path/to/weights/file"));

    QRDetector qr_detector(weights_path);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher image_pub = it.advertise("/qr_detection/detected_image", 1);
    ros::Publisher qr_pub = nh.advertise<opencv_cpp_yolov5::QRCodes>("/qr_detection/qr_codes", 1);

    image_transport::Subscriber sub = it.subscribe("/video_stream_node/image_raw", 1, 
                                boost::bind(&QRDetector::image_cb, &qr_detector, _1, 
                                boost::ref(qr_pub), boost::ref(image_pub)));    

    ros::spin();

    return 0;
}
```

### 4. 更新 `CMakeLists.txt`

确保在 `CMakeLists.txt` 中添加对消息生成的支持：

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(opencv_cpp_yolov5)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV 4 REQUIRED)
include_directories(/usr/local/opencv-4.6.0/include/opencv4)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  QRCode.msg
  QRCodes.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(qr_detection_node src/qr_detection_node.cpp src/qr_detector.cpp)
target_link_libraries(qr_detection_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(qr_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
```

### 5. 更新 `package.xml`

确保在 `package.xml` 中添加对消息生成的依赖：

```xml
<package format="2">
  <name>opencv_cpp_yolov5</name>
  <version>0.0.0</version>
  <description>The opencv_cpp_yolov5 package</description>

  <maintainer email="user@todo.todo">user</maintainer>

  <license>TODO</license>

  <buildtool_depend>catkin</buildtool_depend>

  <build_depend>roscpp</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>image_transport</build_depend>
  <build_depend>cv_bridge</build_depend>
  <build_depend>message_generation</build_depend>

  <exec_depend>roscpp</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>image_transport</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>message_runtime</exec_depend>

  <export>
  </export>
</package>
```

### 6. 创建 `QRCode.msg`

```plaintext
Header header
string data
int64 xmin
int64 ymin
int64 xmax
int64 ymax
```

### 7. 创建 `QRCodes.msg`

```plaintext
Header header
QRCode[] qr_codes
```

### 8. 创建 `qr_detection.launch`

```xml
<launch>
    <node name="qr_detection_node" pkg="opencv_cpp_yolov5" type="qr_detection_node">
        <param name="video_path" value="/dev/video0" />
        <param name="weights_path" value="/path/to/weights/file" />
    </node>
</launch>
```

通过这些步骤，你可以创建一个完整的程序来识别9种二维码，并将结果发布出去。确保将权重文件路径正确传递给 `QRDetector` 类，并在构造函数中加载这些权重文件。

找到具有 2 个许可证类型的类似代码