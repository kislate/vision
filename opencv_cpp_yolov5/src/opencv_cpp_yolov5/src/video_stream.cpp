// Description: Publish video stream to ROS topic
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


// 这个程序用于发布视频流，发布以后可以使用rqt_image_view查看，然后订阅这个话题，进行二维码检测
// 通过修改参数，可以发布不同的视频流   
int main(int argc, char** argv) {
    ros::init(argc, argv, "video_stream_publisher");// 初始化ROS节点
    ros::NodeHandle nh("~"); // 创建节点句柄，句柄的名字是"video_stream_publisher"，~表示私有命名空间，即参数名字前面加上这个名字，如"~video_path"，nh()的意思是查找私有命名空间的参数
    // 公有命名空间的参数名字前面不加这个名字，如"video_path"，nh.param()函数会自动查找参数名字，如果找不到，就使用默认值
    // NodeHandle类的构造函数会初始化节点，析构函数会关闭节点

    // 定义一个字符串变量video_path，用于存放视频文件的路径
    std::string video_path; 

    // 定义两个整型变量frame_height和frame_width，用于存放视频帧的高度和宽度
    int frame_height, frame_width; 

    // 从ROS参数服务器中获取参数，如果没有找到，就使用默认值
    // 可在launch中指定参数，或者yaml文件中指定参数，然后在launch文件中引用yaml文件
    nh.param("video_path", video_path, std::string("/path/to/video.mp4"))
    nh.param("frame_height", frame_height, 480);
    nh.param("frame_width", frame_width, 640);

    // 打印参数信息，显示在终端上
    ROS_INFO("video_path: %s", video_path.c_str());
    ROS_INFO("frame_height: %d", frame_height);
    ROS_INFO("frame_width: %d", frame_width);

    cv::VideoCapture cap(video_path);// 创建一个VideoCapture对象cap，用于读取视频文件
    // VideoCapture类是OpenCV中用于读取视频文件的类，可以从视频文件中读取帧，也可以从摄像头中读取帧
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);

    // .isOpened()函数用于检查视频文件是否打开成功
    if (!cap.isOpened()) {
        // 如果视频文件打开失败，打印错误信息，然后退出程序
        ROS_ERROR("Could not open video file.");
        return 1;
    }// 这就是下面代码拿到的视频文件的路径

    // 创建一个Publisher对象pub，用于发布图像消息
    ros::Publisher pub = nh.advertise<sensor_msgs::Image>("image_raw", 1);
    
    // inmage_raw是图像消息的话题名字，1表示消息队列的长度为1，即最多缓存1个消息，多余的消息会被丢弃
    // 这个图像消息是sensor_msgs/Image类型的，是ROS中的消息类型，用于表示图像数据
    /*
    Header header         # ROS 标准消息头，包含时间戳和帧 ID
    uint32 height         # 图像的高度（行数）
    uint32 width          # 图像的宽度（列数）
    string encoding       # 图像的编码方式（例如 "rgb8", "bgr8", "mono8", "mono16" 等）
    uint8 is_bigendian    # 是否是大端字节序（0: 小端，1: 大端）
    uint32 step           # 每行图像数据的字节数
    uint8[] data          # 实际的图像数据，按行优先顺序存储
    */

    cv::Mat frame;// 创建一个Mat对象frame，用于存放视频帧
    // Mat类是OpenCV中用于存放图像数据的类，可以存放灰度图像、彩色图像、深度图像等
    // Mat类的声明如下：

    sensor_msgs::ImagePtr msg;

    ros::Rate loop_rate(20);  // Adjust the rate as needed
    
    // 循环读取视频文件中的每一帧，然后发布图像消息
    while (nh.ok()) {
        // nh.ok()函数用于检查节点是否正常运行
        cap >> frame;// 从视频文件中读取一帧，存放到Mat对象frame中
        if (!frame.empty()) {
            // 如果读取成功，将Mat对象frame转换为图像消息msg，Header()表示消息头，"bgr8"表示编码方式

            // 之所以一开始不直接使用图像消息，是因为Mat对象更容易处理
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            pub.publish(msg);// 发布图像消息

            cv::waitKey(1);// 等待1毫秒，然后继续读取下一帧
        } else {
            ROS_WARN("End of video file reached. restarting...");

            // 如果读取到视频文件的末尾，将视频文件的位置设置为0，重新开始读取
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }


        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}