#include "qr_detector.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "qr_detection_node");
    ros::NodeHandle nh("~");

    QRDetector qr_detector;

    image_transport::ImageTransport it(nh);
    image_transport::Publisher image_pub = it.advertise("/qr_detection/detected_image", 1);
    ros::Publisher qr_pub = nh.advertise<std_msgs::String>("/qr_detection/qr_data", 1);

    image_transport::Subscriber sub = it.subscribe("/video_stream_node/image_raw", 1, 
                                boost::bind(&QRDetector::image_cb, &qr_detector, _1, 
                                boost::ref(qr_pub), boost::ref(image_pub)));    

    ros::spin();

    return 0;
}