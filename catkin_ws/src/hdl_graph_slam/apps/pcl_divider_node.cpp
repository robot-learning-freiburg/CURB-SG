#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <string>
#include <tuple>
#include <ctime>
#include <thread>
#include <chrono>
#include <random>

typedef pcl::PointXYZINormal PointT;
bool add_noise;

std::tuple <pcl::PointCloud<PointT>::Ptr, pcl::PointCloud<PointT>::Ptr>
process_data(const sensor_msgs::PointCloud2ConstPtr &data, int agent_no) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud <PointT>);
    pcl::fromROSMsg(*data, *cloud);

    pcl::PointCloud<PointT>::Ptr static_objects(new pcl::PointCloud <PointT>);
    pcl::PointCloud<PointT>::Ptr dynamic_objects(new pcl::PointCloud <PointT>);

    std::cout << "cloud size: " << cloud->size() << std::endl;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.02);  // Mean=0, StdDev=0.1



    for (auto &point: cloud->points) {

        if (add_noise) {
            point.x += distribution(generator);
            point.y += distribution(generator);
            point.z += distribution(generator);
        }

        if (static_cast<int>(point.intensity) == 4 || static_cast<int>(point.intensity) == 10) {
            dynamic_objects->push_back(point);
        } else {
            static_objects->push_back(point);
        }
    }

    return std::make_tuple(static_objects, dynamic_objects);
}

void publish_data(const pcl::PointCloud<PointT>::Ptr &static_objects,
                  const pcl::PointCloud<PointT>::Ptr &dynamic_objects,
                  const sensor_msgs::PointCloud2ConstPtr &data, int agent_no, const ros::Publisher &pub_static,
                  const ros::Publisher &pub_dynamic) {

    sensor_msgs::PointCloud2 msg_static;
    sensor_msgs::PointCloud2 msg_dynamic;

    pcl::toROSMsg(*static_objects, msg_static);
    pcl::toROSMsg(*dynamic_objects, msg_dynamic);

    msg_static.header = data->header;
    msg_static.header.frame_id = "base_link_" + std::to_string(agent_no);
    msg_dynamic.header = data->header;
    msg_dynamic.header.frame_id = "base_link_" + std::to_string(agent_no);

    pub_static.publish(msg_static);
    pub_dynamic.publish(msg_dynamic);
}

void raw_pcl_callback(const sensor_msgs::PointCloud2ConstPtr &data, int agent_no, const ros::Publisher &pub_static,
                      const ros::Publisher &pub_dynamic) {

    auto start = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<PointT>::Ptr static_objects, dynamic_objects;
    std::tie(static_objects, dynamic_objects) = process_data(data, agent_no);

    publish_data(static_objects, dynamic_objects, data, agent_no, pub_static, pub_dynamic);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    ROS_INFO("Agent %d - Total Time : %.3f", agent_no, total_time.count());
}

int main(int argc, char **argv) {
    std::cout << "Init start..." << std::endl;
    ros::init(argc, argv, "point_cloud_divider");

    if (argc < 2) {
        ROS_ERROR("Missing argument. Usage: rosrun [your_package] [your_node] [true/false]");
        return 1;
    }

    add_noise = std::string(argv[1]) == "true";

    ros::NodeHandle nh;

    std::vector <ros::Subscriber> subscribers;
    std::vector <ros::Publisher> pub_static_vector;
    std::vector <ros::Publisher> pub_dynamic_vector;


    for (int i = 0; i < 4; ++i) {

        pub_static_vector.push_back(
                nh.advertise<sensor_msgs::PointCloud2>(std::string("/velodyne_points_stat_") + std::to_string(i), 10));
        pub_dynamic_vector.push_back(
                nh.advertise<sensor_msgs::PointCloud2>(std::string("/velodyne_points_dyn_") + std::to_string(i), 10));

    }
    for (int i = 0; i < 4; ++i) {
        subscribers.push_back(
                nh.subscribe<sensor_msgs::PointCloud2>(std::string("/velodyne_points_raw_") + std::to_string(i), 10,
                                                       boost::bind(raw_pcl_callback, _1, i,
                                                                   boost::ref(pub_static_vector[i]),
                                                                   boost::ref(pub_dynamic_vector[i]))));
    }

    std::cout << "Init done." << std::endl;

    ros::spin();
    return 0;
}
