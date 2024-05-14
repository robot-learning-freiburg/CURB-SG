// SPDX-License-Identifier: BSD-2-Clause

#include <string>
#include <hdl_graph_slam/custom_point_types.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_listener.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>


namespace hdl_graph_slam {

    class PrefilteringNodelet : public nodelet::Nodelet {
    public:
        typedef pcl::PointXYZINormal PointT;

        PrefilteringNodelet() {}

        virtual ~PrefilteringNodelet() {}

        virtual void onInit() {
            nh = getNodeHandle();
            private_nh = getPrivateNodeHandle();

            initialize_params();

            if (private_nh.param<bool>("deskewing", false)) {
                imu_sub = nh.subscribe("/imu/data", 1, &PrefilteringNodelet::imu_callback, this);
            }
            agent_no = private_nh.param<std::string>("agent_no", "102");
            points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points_" + agent_no);
            filtered_points_topic = private_nh.param<std::string>("filtered_points_topic",
                                                                  "/filtered_points_" + agent_no);
            colored_points_topic = private_nh.param<std::string>("colored_points_topic", "/colored_points_" + agent_no);

            base_link_frame_id = private_nh.param<std::string>("base_link_frame_id", "");

            points_sub = nh.subscribe(points_topic, 64, &PrefilteringNodelet::cloud_callback, this);
            points_pub = nh.advertise<sensor_msgs::PointCloud2>(filtered_points_topic, 32);
            colored_pub = nh.advertise<sensor_msgs::PointCloud2>(colored_points_topic, 32);

            classes_for_high_resolution = {2, 5, 12, 17, 18, 19, 20};
        }

    private:
        void initialize_params() {
            std::string downsample_method = private_nh.param<std::string>("downsample_method", "VOXELGRID");
            double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);

            if (downsample_method == "VOXELGRID") {
                std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
                auto voxelgrid = new pcl::VoxelGrid<PointT>();
                voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
                downsample_filter.reset(voxelgrid);
            } else if (downsample_method == "APPROX_VOXELGRID") {
                std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
                pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
                approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
                downsample_filter = approx_voxelgrid;
            } else {
                if (downsample_method != "NONE") {
                    std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
                    std::cerr << "       : use passthrough filter" << std::endl;
                }
                std::cout << "downsample: NONE" << std::endl;
            }

            std::string outlier_removal_method = private_nh.param<std::string>("outlier_removal_method", "STATISTICAL");
            if (outlier_removal_method == "STATISTICAL") {
                int mean_k = private_nh.param<int>("statistical_mean_k", 20);
                double stddev_mul_thresh = private_nh.param<double>("statistical_stddev", 1.0);
                std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

                pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
                sor->setMeanK(mean_k);
                sor->setStddevMulThresh(stddev_mul_thresh);
                outlier_removal_filter = sor;
            } else if (outlier_removal_method == "RADIUS") {
                double radius = private_nh.param<double>("radius_radius", 0.8);
                int min_neighbors = private_nh.param<int>("radius_min_neighbors", 2);
                std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

                pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
                rad->setRadiusSearch(radius);
                rad->setMinNeighborsInRadius(min_neighbors);
                outlier_removal_filter = rad;
            } else {
                std::cout << "outlier_removal: NONE" << std::endl;
            }

            use_distance_filter = private_nh.param<bool>("use_distance_filter", true);
            distance_near_thresh = private_nh.param<double>("distance_near_thresh", 1.0);
            distance_far_thresh = private_nh.param<double>("distance_far_thresh", 100.0);

            base_link_frame_id = private_nh.param<std::string>(base_link_frame_id, "");
        }

        void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
            imu_queue.push_back(imu_msg);
        }

        void cloud_callback(const pcl::PointCloud <PointT> &src_cloud_r) {
            pcl::PointCloud<PointT>::ConstPtr src_cloud = src_cloud_r.makeShared();
            if (src_cloud->empty()) {
                std::cout << "EG POINTCLOUD EMPTY" << std::endl;
                return;
            }

            // if base_link_frame is defined, transform the input cloud to the frame
            if (!base_link_frame_id.empty()) {
                if (!tf_listener.canTransform(base_link_frame_id, src_cloud->header.frame_id, ros::Time(0))) {
                    std::cerr << "failed to find transform between " << base_link_frame_id << " and "
                              << src_cloud->header.frame_id << std::endl;
                }

                tf::StampedTransform transform;
                tf_listener.waitForTransform(base_link_frame_id, src_cloud->header.frame_id, ros::Time(0),
                                             ros::Duration(2.0));
                tf_listener.lookupTransform(base_link_frame_id, src_cloud->header.frame_id, ros::Time(0), transform);

                pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
                pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
                transformed->header.frame_id = base_link_frame_id;
                transformed->header.stamp = src_cloud->header.stamp;
                src_cloud = transformed;
            }

            pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);
            filtered = downsample(filtered);
            filtered = outlier_removal(filtered);

            pcl::PointCloud<PointT>::Ptr cloud_of_small_objects(new pcl::PointCloud <PointT>);
            pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());

            for (int i = 0; i < classes_for_high_resolution.size(); i++) {
                range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(
                        new pcl::FieldComparison<PointT>("intensity", pcl::ComparisonOps::EQ,
                                                         classes_for_high_resolution[i])));
            }

            pcl::ConditionalRemoval <PointT> condrem;
            condrem.setCondition(range_cond);
            condrem.setInputCloud(src_cloud);

            // apply filter
            condrem.filter(*cloud_of_small_objects);

            pcl::PointCloud<PointT>::Ptr concatenated_cloud (new pcl::PointCloud<PointT>);
            pcl::concatenate(*filtered, *cloud_of_small_objects, *concatenated_cloud);

            std::cout << "small objects: " << cloud_of_small_objects->points.size() << std::endl;

            points_pub.publish(*concatenated_cloud);
        }

        pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr &cloud) const {
            if (!downsample_filter) {
                return cloud;
            }

            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

            downsample_filter->setInputCloud(cloud);
            downsample_filter->filter(*filtered);
            filtered->header = cloud->header;

            return filtered;
        }

        pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr &cloud) const {
            if (!outlier_removal_filter) {
                return cloud;
            }

            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

            outlier_removal_filter->setInputCloud(cloud);
            outlier_removal_filter->filter(*filtered);
            filtered->header = cloud->header;

            return filtered;
        }

        pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr &cloud) const {
            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
            filtered->reserve(cloud->size());

            std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT &p) {
                double d = p.getVector3fMap().norm();
                return d > distance_near_thresh && d < distance_far_thresh;
            });

            filtered->width = filtered->size();
            filtered->height = 1;
            filtered->is_dense = false;

            filtered->header = cloud->header;

            return filtered;
        }

        pcl::PointCloud<PointT>::ConstPtr deskewing(const pcl::PointCloud<PointT>::ConstPtr &cloud) {
            ros::Time stamp = pcl_conversions::fromPCL(cloud->header.stamp);
            if (imu_queue.empty()) {
                return cloud;
            }


            // the color encodes the point number in the point sequence
            if (colored_pub.getNumSubscribers()) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
                colored->header = cloud->header;
                colored->is_dense = cloud->is_dense;
                colored->width = cloud->width;
                colored->height = cloud->height;
                colored->resize(cloud->size());

                for (int i = 0; i < cloud->size(); i++) {
                    double t = static_cast<double>(i) / cloud->size();
                    colored->at(i).getVector4fMap() = cloud->at(i).getVector4fMap();
                    colored->at(i).r = 255 * t;
                    colored->at(i).g = 128;
                    colored->at(i).b = 255 * (1 - t);
                }
                colored_pub.publish(*colored);
            }

            sensor_msgs::ImuConstPtr imu_msg = imu_queue.front();

            auto loc = imu_queue.begin();
            for (; loc != imu_queue.end(); loc++) {
                imu_msg = (*loc);
                if ((*loc)->header.stamp > stamp) {
                    break;
                }
            }

            imu_queue.erase(imu_queue.begin(), loc);

            Eigen::Vector3f ang_v(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y,
                                  imu_msg->angular_velocity.z);
            ang_v *= -1;

            pcl::PointCloud<PointT>::Ptr deskewed(new pcl::PointCloud<PointT>());
            deskewed->header = cloud->header;
            deskewed->is_dense = cloud->is_dense;
            deskewed->width = cloud->width;
            deskewed->height = cloud->height;
            deskewed->resize(cloud->size());

            double scan_period = private_nh.param<double>("scan_period", 0.1);
            for (int i = 0; i < cloud->size(); i++) {
                const auto &pt = cloud->at(i);

                double delta_t = scan_period * static_cast<double>(i) / cloud->size();
                Eigen::Quaternionf delta_q(1, delta_t / 2.0 * ang_v[0], delta_t / 2.0 * ang_v[1],
                                           delta_t / 2.0 * ang_v[2]);
                Eigen::Vector3f pt_ = delta_q.inverse() * pt.getVector3fMap();

                deskewed->at(i) = cloud->at(i);
                deskewed->at(i).getVector3fMap() = pt_;
            }

            return deskewed;
        }

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        ros::Subscriber imu_sub;
        std::vector <sensor_msgs::ImuConstPtr> imu_queue;

        ros::Subscriber points_sub;
        ros::Publisher points_pub;

        ros::Publisher colored_pub;

        std::vector<int> classes_for_high_resolution;

        tf::TransformListener tf_listener;

        std::string base_link_frame_id;
        std::string colored_points_topic;
        std::string filtered_points_topic;
        std::string agent_no;
        std::string points_topic;

        bool use_distance_filter;
        double distance_near_thresh;
        double distance_far_thresh;

        pcl::Filter<PointT>::Ptr downsample_filter;
        pcl::Filter<PointT>::Ptr outlier_removal_filter;
    };

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::PrefilteringNodelet, nodelet::Nodelet
)
