// SPDX-License-Identifier: BSD-2-Clause

#include <ctime>
#include <mutex>
#include <atomic>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <exception>

#include <ros/ros.h>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include "pcl_ros/transforms.h"
#include "pcl_ros/impl/transforms.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>

#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <hdl_graph_slam/FloorCoeffs.h>

#include <hdl_graph_slam/SaveMap.h>
#include <hdl_graph_slam/DumpGraph.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_graph_slam/ros_utils.hpp>
#include <hdl_graph_slam/ros_time_hash.hpp>
#include <hdl_graph_slam/custom_point_types.hpp>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <hdl_graph_slam/graph_slam.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/dynamic_observation.hpp>
#include <hdl_graph_slam/Keyframe_msg.h>
#include <hdl_graph_slam/DynObservation_msg.h>
#include <hdl_graph_slam/keyframe_updater.hpp>
#include <hdl_graph_slam/loop_detector.hpp>
#include <hdl_graph_slam/information_matrix_calculator.hpp>
#include <hdl_graph_slam/map_cloud_generator.hpp>
#include <hdl_graph_slam/nmea_sentence_parser.hpp>

#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/edge_se3_plane.hpp>
#include <g2o/edge_se3_priorxy.hpp>
#include <g2o/edge_se3_priorxyz.hpp>
#include <g2o/edge_se3_priorvec.hpp>
#include <g2o/edge_se3_priorquat.hpp>

namespace hdl_graph_slam {

    class HdlGraphSlamNodelet : public nodelet::Nodelet {
    public:
        typedef pcl::PointXYZINormal PointT;
        typedef message_filters::sync_policies::ApproximateTime <nav_msgs::Odometry, sensor_msgs::PointCloud2> ApproxSyncPolicy;

        HdlGraphSlamNodelet() {}

        virtual ~HdlGraphSlamNodelet() {}

        virtual void onInit() {
            nh = getNodeHandle();
            mt_nh = getMTNodeHandle();
            private_nh = getPrivateNodeHandle();

            // init parameters
            agent_no = private_nh.param<std::string>("agent_no", "101");
            map_frame_id = private_nh.param<std::string>("map_frame_id", "map_" + agent_no);
            keyframe_frame_id = private_nh.param<std::string>("keyframe_frame_id", "keyframe_" + agent_no);
            agent_position_topic = private_nh.param<std::string>("agent_position_topic", "agent_position_" + agent_no);
            dyn_points_topic = private_nh.param<std::string>("dyn_points_topic", "/velodyne_points_dyn_" + agent_no);
            agent_keyframe_topic = private_nh.param<std::string>("agent_keyframe_topic", "/agent_keyframes");
            dynamic_object_topic = private_nh.param<std::string>("dynamic_object_topic", "/dynamic_object_keyframes");
            dynamic_observation_topic = private_nh.param<std::string>("dynamic_observation_topic",
                                                                      "/dynamic_observation_" + agent_no);
            odom_frame_id = private_nh.param<std::string>("odom_frame_id", "odom_" + agent_no);
            filtered_points_topic = private_nh.param<std::string>("filtered_points_topic",
                                                                  "/filtered_points_" + agent_no);
            hdl_graph_slam_topic = private_nh.param<std::string>("hdl_graph_slam_topic", "/hdl_graph_slam_" + agent_no);
            floor_detection_topic = private_nh.param<std::string>("floor_detection_topic",
                                                                  "/floor_detection_" + agent_no);
            pose_topic = private_nh.param<std::string>("pose_topic", "/pose_" + agent_no);
            initial_pose = Eigen::Isometry3d::Identity();
            initial_yaw = 0;
            skipping_obs = false;
            act_pose = Eigen::Isometry3d::Identity();

            setInitialPosition = true;
            setInitialPosition_orientation = true;
            setInitialPosition_keyframe = true;
            indexSendKeyframes = 0;

            keyframe_id = stoi(agent_no) * 1000000;
            observation_id = stoi(agent_no) * 1000000;

            tf2_ros::Buffer tfBuffer;

            map_cloud_resolution = private_nh.param<double>("map_cloud_resolution", 0.05);
            min_points_for_detection = private_nh.param<double>("min_points_for_detection", 1000);
            trans_odom2map.setIdentity();

            max_keyframes_per_update = private_nh.param<int>("max_keyframes_per_update", 10);

            anchor_node = nullptr;
            anchor_edge = nullptr;
            floor_plane_node = nullptr;
            graph_slam.reset(new GraphSLAM(private_nh.param<std::string>("g2o_solver_type", "lm_var")));
            keyframe_updater.reset(new KeyframeUpdater(private_nh));
            loop_detector.reset(new LoopDetector(private_nh));
            map_cloud_generator.reset(new MapCloudGenerator());
            inf_calclator.reset(new InformationMatrixCalculator(private_nh));

            floor_edge_stddev = private_nh.param<double>("floor_edge_stddev", 10.0);

            // subscribers
            odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, odom_frame_id, 256));
            cloud_sub.reset(
                    new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, filtered_points_topic, 32));
            sync.reset(
                    new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(32), *odom_sub, *cloud_sub));
            sync->registerCallback(boost::bind(&HdlGraphSlamNodelet::cloud_callback, this, _1, _2));

            floor_sub = nh.subscribe(floor_detection_topic + "/floor_coeffs", 1024,
                                     &HdlGraphSlamNodelet::floor_coeffs_callback, this);
            pose_sub = nh.subscribe(pose_topic, 1024, &HdlGraphSlamNodelet::pose_callback, this);
            dyn_sub = nh.subscribe(dyn_points_topic, 64, &HdlGraphSlamNodelet::dyn_cloud_callback, this);
            command_sub = nh.subscribe(hdl_graph_slam_topic + "/command", 1024, &HdlGraphSlamNodelet::command_callback,
                                       this);

            // publishers
            markers_pub = mt_nh.advertise<visualization_msgs::MarkerArray>(hdl_graph_slam_topic + "/markers", 16);
            odom2map_pub = mt_nh.advertise<geometry_msgs::TransformStamped>(hdl_graph_slam_topic + "/odom2map", 16);
            map_points_pub = mt_nh.advertise<sensor_msgs::PointCloud2>(hdl_graph_slam_topic + "/map_points", 1, true);
            read_until_pub = mt_nh.advertise<std_msgs::Header>(hdl_graph_slam_topic + "/read_until", 32);
            keyframe_pub = mt_nh.advertise<hdl_graph_slam::Keyframe_msg>(agent_keyframe_topic, 64);
            dynamic_observation_pub = mt_nh.advertise<hdl_graph_slam::DynObservation_msg>(dynamic_observation_topic,
                                                                                          64);
            dynamic_object_box_pub = mt_nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(
                    "/dyn_boxes_agent_" + agent_no, 64);

            dump_service_server = mt_nh.advertiseService(hdl_graph_slam_topic + "/dump",
                                                         &HdlGraphSlamNodelet::dump_service, this);
            save_map_service_server = mt_nh.advertiseService(hdl_graph_slam_topic + "/save_map",
                                                             &HdlGraphSlamNodelet::save_map_service, this);

            graph_updated = false;
            double graph_update_interval = private_nh.param<double>("graph_update_interval", 3.0);
            double map_cloud_update_interval = private_nh.param<double>("map_cloud_update_interval", 10.0);
            optimization_timer = mt_nh.createWallTimer(ros::WallDuration(graph_update_interval),
                                                       &HdlGraphSlamNodelet::optimization_timer_callback, this);
            map_publish_timer = mt_nh.createWallTimer(ros::WallDuration(map_cloud_update_interval),
                                                      &HdlGraphSlamNodelet::map_points_publish_timer_callback, this);
        }

    private:
        /**
         * @brief received point clouds are pushed to #keyframe_queue
         * @param odom_msg
         * @param cloud_msg
         */
        void cloud_callback(const nav_msgs::OdometryConstPtr &odom_msg,
                            const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) {

            const ros::Time &stamp = cloud_msg->header.stamp;
            Eigen::Isometry3d odom = odom2isometry(odom_msg);

            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::fromROSMsg(*cloud_msg, *cloud);

            std::cout << "size of cloud: " << cloud->points.size() << std::endl;
            std::cout << "keyframe stamp: " << stamp << std::endl;

            if (base_frame_id.empty()) {
                base_frame_id = cloud_msg->header.frame_id;
            }

            if (!keyframe_updater->update(odom)) {
                std::lock_guard <std::mutex> lock(keyframe_queue_mutex);
                if (keyframe_queue.empty()) {
                    std_msgs::Header read_until;
                    read_until.stamp = stamp + ros::Duration(10, 0);
                    read_until.frame_id = points_topic;
                    read_until_pub.publish(read_until);
                    read_until.frame_id = filtered_points_topic;
                    read_until_pub.publish(read_until);
                }
                return;
            }

            double accum_d = keyframe_updater->get_accum_distance();
            KeyFrame::Ptr keyframe(new KeyFrame(stamp, keyframe_id++, odom, accum_d, cloud));


            std::cout << "created keyframe with ID: " << keyframe_id << std::endl;

            std::lock_guard <std::mutex> lock(keyframe_queue_mutex);
            keyframe_queue.push_back(keyframe);
        }

        void createDynamicObjectBoundingBoxes(ros::Time newest_stamp) {
            jsk_recognition_msgs::BoundingBoxArray boxArray;
            boxArray.header.frame_id = keyframe_frame_id;

            boxArray.header.stamp = newest_stamp;
            boxArray.boxes.clear();

            int boxNumber = 0;

            std::cout << std::endl;
            std::cout << "Dynamic Object Queue: " << dynamic_object_queue.size() << std::endl;
            std::cout << std::endl;

            for (const auto &obj: dynamic_object_queue) {
                boxNumber++;
                jsk_recognition_msgs::BoundingBox box;

                box.header = boxArray.header;
                box.pose.orientation.w = 1;

                box.label = stoi(agent_no);
                box.value = stoi(agent_no) * 1.0;

                box.dimensions.x = 2.0;
                box.dimensions.y = 2.0;
                box.dimensions.z = 1.0;

                box.pose.position.x = obj->transformFromKeyframe.position.x;
                box.pose.position.y = obj->transformFromKeyframe.position.y;
                box.pose.position.z = 1.0;

                boxArray.boxes.push_back(box);
            }
            dynamic_object_box_pub.publish(boxArray);

            std::cout << "box array published with " << boxNumber << " boxes" << std::endl;
        }


        /**
         * @brief received point clouds are pushed to #keyframe_queue
         * @param odom_msg
         * @param cloud_msg
         */
        void dyn_cloud_callback(const sensor_msgs::PointCloud2::ConstPtr &dyn_cloud_msg) {

            flush_keyframe_queue();
            optimization();


            if (keyframes.empty()) {
                std::cout << "Dyn Cloud return: no keyframes yet." << std::endl;
                return;
            }

            std::cout << std::endl;

            const ros::Time &stamp = dyn_cloud_msg->header.stamp;
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr dyn_cloud_msg_out(new pcl::PointCloud<PointT>());
            pcl::fromROSMsg(*dyn_cloud_msg, *cloud);

            cloud->points.resize(cloud->width * cloud->height);
            std::list<int> contained_ids = {};

            for (const auto &point: *cloud) {
                if ((std::find(contained_ids.begin(), contained_ids.end(), int(point.curvature)) ==
                     contained_ids.end()) && int(point.curvature) > 0) {
                    contained_ids.push_back(int(point.curvature));
                }
            }

            if (contained_ids.size() == 0) {
                std::cout << "No Dynamic Objects registered. " << std::endl;
                return;
            } else {
                std::cout << "contained objects: " << contained_ids.size() << std::endl;
            }

            pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
            pcl::PassThrough <PointT> pass;
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("curvature");

            for (int id: contained_ids) {
                pass.setFilterLimits(id, id);

                pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
                cloud_filtered->clear();

                pass.filter(*cloud_filtered);

                if (cloud_filtered->size() < min_points_for_detection) {
                    continue;
                }

                float x_min = 1000.0;
                float x_max = -1000.0;
                float y_min = 1000.0;
                float y_max = -1000.0;

                int object_tag = 0;

                for (const auto &point: cloud_filtered->points) {
                    x_min = std::min(x_min, point.x);
                    x_max = std::max(x_max, point.x);
                    y_min = std::min(y_min, point.y);
                    y_max = std::max(y_max, point.y);
                    object_tag = point.intensity;
                }

                float x_pos = (x_max - x_min) / 2 + x_min;
                float y_pos = (y_max - y_min) / 2 + y_min;

                geometry_msgs::PoseStamped ps_direct;
                geometry_msgs::PoseStamped ps_transformed;

                ps_direct.pose.position.x = x_pos;
                ps_direct.pose.position.y = y_pos;
                ps_direct.pose.position.z = 0.0;
                ps_direct.pose.orientation.w = 1.0;

                ps_direct.header.frame_id = agent_position_topic;
                geometry_msgs::Pose psout;

                try {
                    tf::StampedTransform transform;
                    tf_listener.waitForTransform(keyframe_frame_id, agent_position_topic, stamp, ros::Duration(5.0));
                    tf_listener.lookupTransform(keyframe_frame_id, agent_position_topic, stamp, transform);

                    // Extract the rotation quaternion from the transform
                    tf::Quaternion quat = transform.getRotation();

                    // Convert the quaternion to Euler angles and extract the yaw angle
                    double yaw = tf::getYaw(quat);

                    if (std::abs(yaw) > 0.02) {

                        ROS_INFO("Skipped dyn-obs due to high turning angle: yaw = %f", std::abs(yaw));

                        skipping_obs = true;

                        continue;
                    }

                    if (skipping_obs) {
                        // skip one more time, to handle outliers in the yaw value
                        skipping_obs = false;
                        continue;
                    }

                    tf_listener.transformPose(keyframe_frame_id, ps_direct, ps_transformed);
                    psout = ps_transformed.pose;
                }
                catch (tf::TransformException &ex) {
                    std::cout << "box transform failed" << std::endl;
                    std::cout << "from " << agent_position_topic << " to " << keyframe_frame_id << std::endl;
                    ROS_ERROR("Received an exception trying to transform a point: %s",
                              ex.what());
                }


                try {
                    DynObservation::Ptr obs(
                            new DynObservation(
                                    observation_id++,
                                    stamp,
                                    keyframes.back()->keyframe_id,
                                    id,
                                    object_tag,
                                    psout
                            ));

                    dynamic_object_queue.push_back(obs);
                }
                catch (std::exception &e) {
                    std::cout << e.what() << std::endl;
                }
            }
        }

        void command_callback(const std_msgs::String::ConstPtr &msg) {
            std::string command = msg->data.c_str();
            if (!command.compare("save_graph")) {
                if (keyframes.empty()) {
                    ROS_INFO("No Keyframes yet.");
                    return;
                }
                sendKeyframesToServer();
            } else if (!command.compare("pub_graph")) {
                ROS_INFO("Publishing graph");
            }
        }

        void sendKeyframesToServer() {
            std::vector <KeyFrame::Ptr> send_keyframes;

            for (int i = indexSendKeyframes; i < keyframes.size(); i++) {
                send_keyframes.push_back(keyframes[i]);
            }

            ROS_INFO("Sending keyframes - Index: %i/%li - sending: %li Keyframes", indexSendKeyframes,
                     keyframes.size(), send_keyframes.size());

            // save keyframes beside the graph
            for (const auto &keyframe: send_keyframes) {
                hdl_graph_slam::Keyframe_msg msg;

                msg.header.stamp = keyframe->stamp;
                msg.header.frame_id = "/odom";
                msg.accum_distance = keyframe->accum_distance;
                sensor_msgs::PointCloud2 cloud_msg;
                pcl::toROSMsg(*keyframe->cloud, cloud_msg);
                msg.cloud = cloud_msg;

                geometry_msgs::Pose odom_msg;

                odom_msg.position.x = keyframe->odom.translation()[0];
                odom_msg.position.y = keyframe->odom.translation()[1];
                odom_msg.position.z = keyframe->odom.translation()[2];
                Eigen::Quaterniond q = (Eigen::Quaterniond) keyframe->odom.linear();
                odom_msg.orientation.x = q.x();
                odom_msg.orientation.y = q.y();
                odom_msg.orientation.z = q.z();
                odom_msg.orientation.w = q.w();
                if (odom_msg.orientation.w < 0) {
                    odom_msg.orientation.x *= -1;
                    odom_msg.orientation.y *= -1;
                    odom_msg.orientation.z *= -1;
                    odom_msg.orientation.w *= -1;
                }
                msg.odom = odom_msg;

                if (indexSendKeyframes == 0) {
                    geometry_msgs::Pose initial_pose_msg;

                    initial_pose_msg.position.x = initial_pose.translation()[0];
                    initial_pose_msg.position.y = initial_pose.translation()[1];
                    initial_pose_msg.position.z = initial_pose.translation()[2];
                    initial_pose_msg.orientation.x = 0;
                    initial_pose_msg.orientation.y = 0;
                    initial_pose_msg.orientation.z = initial_yaw;
                    initial_pose_msg.orientation.w = 0;

                    msg.initial = initial_pose_msg;
                }

                msg.id = keyframe->keyframe_id;

                msg.agent_no = stoi(agent_no);

                keyframe_pub.publish(msg);
                indexSendKeyframes++;
            }
            send_keyframes.clear();
        }

        void sendDynamicObservationsToServer() {

            ROS_INFO("Sending %li dynamic observations", dynamic_object_queue.size());
            std::cout << std::endl;

            // save keyframes beside the graph
            for (const auto &obs: dynamic_object_queue) {
                hdl_graph_slam::DynObservation_msg msg;

                msg.header.stamp = ros::Time::now();
                msg.header.frame_id = "/keyframe_" + agent_no;

                msg.agent_no = stoi(agent_no);
                msg.id = obs->id;
                msg.obs_stamp = obs->stamp;
                msg.keyframe_id = obs->keyframe_id;
                msg.vehicle_id = obs->vehicle_id;
                msg.object_tag = obs->object_tag;
                msg.transform = obs->transformFromKeyframe;

                std::cout << ".";
                dynamic_observation_pub.publish(msg);
            }
            std::cout << std::endl;
            std::cout << "clean object queue" << std::endl;
            dynamic_object_queue.clear();
        }

        void poseEigenToMsg(const Eigen::Isometry3d &e, geometry_msgs::Pose &m) {
            m.position.x = e.translation()[0];
            m.position.y = e.translation()[1];
            m.position.z = e.translation()[2];
            Eigen::Quaterniond q = (Eigen::Quaterniond) e.linear();
            m.orientation.x = q.x();
            m.orientation.y = q.y();
            m.orientation.z = q.z();
            m.orientation.w = q.w();
            if (m.orientation.w < 0) {
                m.orientation.x *= -1;
                m.orientation.y *= -1;
                m.orientation.z *= -1;
                m.orientation.w *= -1;
            }
        }

        void vectorEigenToMsg(const Eigen::Vector3d &e, geometry_msgs::Vector3 &m) {
            m.x = e(0);
            m.y = e(1);
            m.z = e(2);
        }

        /**
         * @brief this method adds all the keyframes in #keyframe_queue to the pose graph (odometry edges)
         * @return if true, at least one keyframe was added to the pose graph
         */
        bool flush_keyframe_queue() {
            std::lock_guard <std::mutex> lock(keyframe_queue_mutex);

            if (keyframe_queue.empty()) {
                return false;
            }

            trans_odom2map_mutex.lock();
            Eigen::Isometry3d odom2map(trans_odom2map.cast<double>());
            trans_odom2map_mutex.unlock();

            int num_processed = 0;
            for (int i = 0; i < std::min<int>(keyframe_queue.size(), max_keyframes_per_update); i++) {
                num_processed = i;

                const auto &keyframe = keyframe_queue[i];
                // new_keyframes will be tested later for loop closure
                new_keyframes.push_back(keyframe);

                ROS_INFO("New keyframe.");

                // add pose node
                Eigen::Isometry3d odom = odom2map * keyframe->odom;

                // if this is the first time visiting this place,
                // set the first node to the initial pose and fix this one.
                if (setInitialPosition) {
                    keyframe->node = graph_slam->add_se3_node(initial_pose);
                    keyframe->node->setFixed(true);
                    setInitialPosition = false;
                } else {
                    keyframe->node = graph_slam->add_se3_node(odom);
                }

                keyframe_hash[keyframe->stamp] = keyframe;

                if (i == 0 && keyframes.empty()) {
                    continue;
                }

                // add edge between consecutive keyframes
                const auto &prev_keyframe = i == 0 ? keyframes.back() : keyframe_queue[i - 1];

                Eigen::Isometry3d relative_pose = keyframe->odom.inverse() * prev_keyframe->odom;

                Eigen::MatrixXd information = inf_calclator->calc_information_matrix(keyframe->cloud,
                                                                                     prev_keyframe->cloud,
                                                                                     relative_pose);
                auto edge = graph_slam->add_se3_edge(keyframe->node, prev_keyframe->node, relative_pose, information);
                graph_slam->add_robust_kernel(edge,
                                              private_nh.param<std::string>("odometry_edge_robust_kernel", "NONE"),
                                              private_nh.param<double>("odometry_edge_robust_kernel_size", 1.0));

            }

            std_msgs::Header read_until;
            read_until.stamp = keyframe_queue[num_processed]->stamp + ros::Duration(10, 0);
            read_until.frame_id = points_topic;
            read_until_pub.publish(read_until);
            read_until.frame_id = filtered_points_topic;
            read_until_pub.publish(read_until);

            keyframe_queue.erase(keyframe_queue.begin(), keyframe_queue.begin() + num_processed + 1);
            return true;
        }

        /**
       * this method is the callback for the initial pose
       */
        void pose_callback(const geometry_msgs::PoseStamped::ConstPtr &pose_msg) {
            if (!(initial_pose.translation()[0] == 0.0 && initial_pose.translation()[1] == 0.0)) {
                return;
            }

            initial_pose.translation().x() = pose_msg->pose.position.x;
            initial_pose.translation().y() = pose_msg->pose.position.y;
            initial_pose.translation().z() = pose_msg->pose.position.z;

            Eigen::Quaterniond q;
            q = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(pose_msg->pose.orientation.z, Eigen::Vector3d::UnitZ());

            initial_orientation = q;
            initial_pose.rotate(q.matrix());

            initial_yaw = pose_msg->pose.orientation.z;

            ROS_INFO("HDL: Initial Pose received and set");
        }

        /**
         * @brief received floor coefficients are added to #floor_coeffs_queue
         * @param floor_coeffs_msg
         */
        void floor_coeffs_callback(const hdl_graph_slam::FloorCoeffsConstPtr &floor_coeffs_msg) {
            if (floor_coeffs_msg->coeffs.empty()) {
                return;
            }

            std::lock_guard <std::mutex> lock(floor_coeffs_queue_mutex);
            floor_coeffs_queue.push_back(floor_coeffs_msg);
        }

        /**
         * @brief this methods associates floor coefficients messages with registered keyframes, and then adds the associated coeffs to the pose graph
         * @return if true, at least one floor plane edge is added to the pose graph
         */
        bool flush_floor_queue() {
            std::lock_guard <std::mutex> lock(floor_coeffs_queue_mutex);

            if (keyframes.empty()) {
                return false;
            }

            const auto &latest_keyframe_stamp = keyframes.back()->stamp;

            bool updated = false;
            for (const auto &floor_coeffs: floor_coeffs_queue) {
                if (floor_coeffs->header.stamp > latest_keyframe_stamp) {
                    break;
                }

                auto found = keyframe_hash.find(floor_coeffs->header.stamp);
                if (found == keyframe_hash.end()) {
                    continue;
                }

                if (!floor_plane_node) {
                    floor_plane_node = graph_slam->add_plane_node(Eigen::Vector4d(0.0, 0.0, 1.0, 0.0));
                    floor_plane_node->setFixed(true);
                }

                const auto &keyframe = found->second;

                Eigen::Vector4d coeffs(floor_coeffs->coeffs[0], floor_coeffs->coeffs[1], floor_coeffs->coeffs[2],
                                       floor_coeffs->coeffs[3]);
                Eigen::Matrix3d information = Eigen::Matrix3d::Identity() * (1.0 / floor_edge_stddev);
                auto edge = graph_slam->add_se3_plane_edge(keyframe->node, floor_plane_node, coeffs, information);
                graph_slam->add_robust_kernel(edge, private_nh.param<std::string>("floor_edge_robust_kernel", "NONE"),
                                              private_nh.param<double>("floor_edge_robust_kernel_size", 1.0));

                keyframe->floor_coeffs = coeffs;

                updated = true;

                std::cout << " ################### " << std::endl;
                std::cout << " floor coeffs inserted in graph " << std::endl;
                std::cout << " ################### " << std::endl;
            }

            auto remove_loc = std::upper_bound(floor_coeffs_queue.begin(), floor_coeffs_queue.end(),
                                               latest_keyframe_stamp, [=](const ros::Time &stamp,
                                                                          const hdl_graph_slam::FloorCoeffsConstPtr &coeffs) {
                        return stamp < coeffs->header.stamp;
                    });
            floor_coeffs_queue.erase(floor_coeffs_queue.begin(), remove_loc);

            return updated;
        }

        /**
         * @brief generate map point cloud and publish it
         * @param event
         */
        void map_points_publish_timer_callback(const ros::WallTimerEvent &event) {
            // we dont need the map of the agents, so we skip the generation
            // to free computation resouces
            return;
        }

        /**
         * @brief this methods adds all the data in the queues to the pose graph, and then optimizes the pose graph
         * @param event
         */
        void optimization_timer_callback(const ros::WallTimerEvent &event) {
            optimization();
        }
        void optimization() {
            std::lock_guard <std::mutex> lock(main_thread_mutex);

            // add keyframes and floor coeffs in the queues to the pose graph
            bool keyframe_updated = flush_keyframe_queue();

            if (!keyframe_updated) {
                std_msgs::Header read_until;
                read_until.stamp = ros::Time::now() + ros::Duration(30, 0);
                read_until.frame_id = points_topic;
                read_until_pub.publish(read_until);
                read_until.frame_id = filtered_points_topic;
                read_until_pub.publish(read_until);
            }

            std::copy(new_keyframes.begin(), new_keyframes.end(), std::back_inserter(keyframes));
            new_keyframes.clear();

            // move the first node anchor position to the current estimate of the first node pose
            // so the first node moves freely while trying to stay around the origin
            if (anchor_node && private_nh.param<bool>("fix_first_node_adaptive", true)) {
                Eigen::Isometry3d anchor_target = static_cast<g2o::VertexSE3 *>(anchor_edge->vertices()[1])->estimate();
                anchor_node->setEstimate(anchor_target);
            }

            // optimize the pose graph
            if (keyframes.size() == 0) {
                // no keyframes receives yet, so nothing to publish
                return;
            }

            // publish tf
            const auto &keyframe = keyframes.back();

            Eigen::Isometry3d trans = keyframe->node->estimate();
            trans_odom2map_mutex.lock();
            trans_odom2map = trans.matrix().cast<float>();
            trans_odom2map_mutex.unlock();

            std::vector <KeyFrameSnapshot::Ptr> snapshot(keyframes.size());
            std::transform(keyframes.begin(), keyframes.end(), snapshot.begin(),
                           [=](const KeyFrame::Ptr &k) { return std::make_shared<KeyFrameSnapshot>(k); });

            keyframes_snapshot_mutex.lock();
            keyframes_snapshot.swap(snapshot);
            keyframes_snapshot_mutex.unlock();
            graph_updated = true;

            if (odom2map_pub.getNumSubscribers()) {
                geometry_msgs::TransformStamped ts = matrix2transform(keyframe->stamp, trans_odom2map,
                                                                      map_frame_id, odom_frame_id);
                odom2map_pub.publish(ts);
            }

            if (markers_pub.getNumSubscribers()) {
                auto markers = create_marker_array(ros::Time::now());
                markers_pub.publish(markers);
            }

            sendKeyframesToServer();
            sendDynamicObservationsToServer();
        }

        /**
         * @brief create visualization marker
         * @param stamp
         * @return
         */
        visualization_msgs::MarkerArray create_marker_array(const ros::Time &stamp) const {
            visualization_msgs::MarkerArray markers;
            markers.markers.resize(4);

            // node markers
            visualization_msgs::Marker &traj_marker = markers.markers[0];
            traj_marker.header.frame_id = map_frame_id;
            traj_marker.header.stamp = stamp;
            traj_marker.ns = "nodes";
            traj_marker.id = 0;
            traj_marker.type = visualization_msgs::Marker::SPHERE_LIST;

            traj_marker.pose.orientation.w = 1.0;
            traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.5;

            visualization_msgs::Marker &imu_marker = markers.markers[1];
            imu_marker.header = traj_marker.header;
            imu_marker.ns = "imu";
            imu_marker.id = 1;
            imu_marker.type = visualization_msgs::Marker::SPHERE_LIST;

            imu_marker.pose.orientation.w = 1.0;
            imu_marker.scale.x = imu_marker.scale.y = imu_marker.scale.z = 0.75;

            traj_marker.points.resize(keyframes.size());
            traj_marker.colors.resize(keyframes.size());
            for (int i = 0; i < keyframes.size(); i++) {
                Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
                traj_marker.points[i].x = pos.x();
                traj_marker.points[i].y = pos.y();
                traj_marker.points[i].z = pos.z();

                double p = static_cast<double>(i) / keyframes.size();
                traj_marker.colors[i].r = 1.0 - p;
                traj_marker.colors[i].g = p;
                traj_marker.colors[i].b = 0.0;
                traj_marker.colors[i].a = 1.0;

                if (keyframes[i]->acceleration) {
                    Eigen::Vector3d pos = keyframes[i]->node->estimate().translation();
                    geometry_msgs::Point point;
                    point.x = pos.x();
                    point.y = pos.y();
                    point.z = pos.z();

                    std_msgs::ColorRGBA color;
                    color.r = 0.0;
                    color.g = 0.0;
                    color.b = 1.0;
                    color.a = 0.1;

                    imu_marker.points.push_back(point);
                    imu_marker.colors.push_back(color);
                }
            }

            // edge markers
            visualization_msgs::Marker &edge_marker = markers.markers[2];
            edge_marker.header.frame_id = map_frame_id;
            edge_marker.header.stamp = stamp;
            edge_marker.ns = "edges";
            edge_marker.id = 2;
            edge_marker.type = visualization_msgs::Marker::LINE_LIST;

            edge_marker.pose.orientation.w = 1.0;
            edge_marker.scale.x = 0.05;

            edge_marker.points.resize(graph_slam->graph->edges().size() * 2);
            edge_marker.colors.resize(graph_slam->graph->edges().size() * 2);

            auto edge_itr = graph_slam->graph->edges().begin();
            for (int i = 0; edge_itr != graph_slam->graph->edges().end(); edge_itr++, i++) {
                g2o::HyperGraph::Edge *edge = *edge_itr;
                g2o::EdgeSE3 *edge_se3 = dynamic_cast<g2o::EdgeSE3 *>(edge);
                if (edge_se3) {
                    g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge_se3->vertices()[0]);
                    g2o::VertexSE3 *v2 = dynamic_cast<g2o::VertexSE3 *>(edge_se3->vertices()[1]);
                    Eigen::Vector3d pt1 = v1->estimate().translation();
                    Eigen::Vector3d pt2 = v2->estimate().translation();

                    edge_marker.points[i * 2].x = pt1.x();
                    edge_marker.points[i * 2].y = pt1.y();
                    edge_marker.points[i * 2].z = pt1.z();
                    edge_marker.points[i * 2 + 1].x = pt2.x();
                    edge_marker.points[i * 2 + 1].y = pt2.y();
                    edge_marker.points[i * 2 + 1].z = pt2.z();

                    double p1 = static_cast<double>(v1->id()) / graph_slam->graph->vertices().size();
                    double p2 = static_cast<double>(v2->id()) / graph_slam->graph->vertices().size();
                    edge_marker.colors[i * 2].r = 1.0 - p1;
                    edge_marker.colors[i * 2].g = p1;
                    edge_marker.colors[i * 2].a = 1.0;
                    edge_marker.colors[i * 2 + 1].r = 1.0 - p2;
                    edge_marker.colors[i * 2 + 1].g = p2;
                    edge_marker.colors[i * 2 + 1].a = 1.0;

                    if (std::abs(v1->id() - v2->id()) > 2) {
                        edge_marker.points[i * 2].z += 0.5;
                        edge_marker.points[i * 2 + 1].z += 0.5;
                    }

                    continue;
                }

                g2o::EdgeSE3Plane *edge_plane = dynamic_cast<g2o::EdgeSE3Plane *>(edge);
                if (edge_plane) {
                    g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge_plane->vertices()[0]);
                    Eigen::Vector3d pt1 = v1->estimate().translation();
                    Eigen::Vector3d pt2(pt1.x(), pt1.y(), 0.0);

                    edge_marker.points[i * 2].x = pt1.x();
                    edge_marker.points[i * 2].y = pt1.y();
                    edge_marker.points[i * 2].z = pt1.z();
                    edge_marker.points[i * 2 + 1].x = pt2.x();
                    edge_marker.points[i * 2 + 1].y = pt2.y();
                    edge_marker.points[i * 2 + 1].z = pt2.z();

                    edge_marker.colors[i * 2].b = 1.0;
                    edge_marker.colors[i * 2].a = 1.0;
                    edge_marker.colors[i * 2 + 1].b = 1.0;
                    edge_marker.colors[i * 2 + 1].a = 1.0;

                    continue;
                }

                g2o::EdgeSE3PriorXY *edge_priori_xy = dynamic_cast<g2o::EdgeSE3PriorXY *>(edge);
                if (edge_priori_xy) {
                    g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge_priori_xy->vertices()[0]);
                    Eigen::Vector3d pt1 = v1->estimate().translation();
                    Eigen::Vector3d pt2 = Eigen::Vector3d::Zero();
                    pt2.head<2>() = edge_priori_xy->measurement();

                    edge_marker.points[i * 2].x = pt1.x();
                    edge_marker.points[i * 2].y = pt1.y();
                    edge_marker.points[i * 2].z = pt1.z() + 0.5;
                    edge_marker.points[i * 2 + 1].x = pt2.x();
                    edge_marker.points[i * 2 + 1].y = pt2.y();
                    edge_marker.points[i * 2 + 1].z = pt2.z() + 0.5;

                    edge_marker.colors[i * 2].r = 1.0;
                    edge_marker.colors[i * 2].a = 1.0;
                    edge_marker.colors[i * 2 + 1].r = 1.0;
                    edge_marker.colors[i * 2 + 1].a = 1.0;

                    continue;
                }

                g2o::EdgeSE3PriorXYZ *edge_priori_xyz = dynamic_cast<g2o::EdgeSE3PriorXYZ *>(edge);
                if (edge_priori_xyz) {
                    g2o::VertexSE3 *v1 = dynamic_cast<g2o::VertexSE3 *>(edge_priori_xyz->vertices()[0]);
                    Eigen::Vector3d pt1 = v1->estimate().translation();
                    Eigen::Vector3d pt2 = edge_priori_xyz->measurement();

                    edge_marker.points[i * 2].x = pt1.x();
                    edge_marker.points[i * 2].y = pt1.y();
                    edge_marker.points[i * 2].z = pt1.z() + 0.5;
                    edge_marker.points[i * 2 + 1].x = pt2.x();
                    edge_marker.points[i * 2 + 1].y = pt2.y();
                    edge_marker.points[i * 2 + 1].z = pt2.z();

                    edge_marker.colors[i * 2].r = 1.0;
                    edge_marker.colors[i * 2].a = 1.0;
                    edge_marker.colors[i * 2 + 1].r = 1.0;
                    edge_marker.colors[i * 2 + 1].a = 1.0;

                    continue;
                }
            }

            // sphere
            visualization_msgs::Marker &sphere_marker = markers.markers[3];
            sphere_marker.header.frame_id = map_frame_id;
            sphere_marker.header.stamp = stamp;
            sphere_marker.ns = "loop_close_radius";
            sphere_marker.id = 3;
            sphere_marker.type = visualization_msgs::Marker::SPHERE;

            if (!keyframes.empty()) {
                Eigen::Vector3d pos = keyframes.back()->node->estimate().translation();
                sphere_marker.pose.position.x = pos.x();
                sphere_marker.pose.position.y = pos.y();
                sphere_marker.pose.position.z = pos.z();
            }
            sphere_marker.pose.orientation.w = 1.0;
            sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z =
                    loop_detector->get_distance_thresh() * 2.0;

            sphere_marker.color.r = 1.0;
            sphere_marker.color.a = 0.3;

            return markers;
        }

        /**
         * @brief dump all data to the current directory
         * @param req
         * @param res
         * @return
         */
        bool dump_service(hdl_graph_slam::DumpGraphRequest &req, hdl_graph_slam::DumpGraphResponse &res) {
            std::lock_guard <std::mutex> lock(main_thread_mutex);

            std::string directory = req.destination;

            if (directory.empty()) {
                std::array<char, 64> buffer;
                buffer.fill(0);
                time_t rawtime;
                time(&rawtime);
                const auto timeinfo = localtime(&rawtime);
                strftime(buffer.data(), sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
            }

            if (!boost::filesystem::is_directory(directory)) {
                boost::filesystem::create_directory(directory);
            }

            std::cout << "all data dumped to:" << directory << std::endl;

            graph_slam->save(directory + "/graph.g2o");
            for (int i = 0; i < keyframes.size(); i++) {
                std::stringstream sst;
                sst << boost::format("%s/%06d") % directory % i;

                keyframes[i]->save(sst.str());
            }

            if (zero_utm) {
                std::ofstream zero_utm_ofs(directory + "/zero_utm");
                zero_utm_ofs << boost::format("%.6f %.6f %.6f") % zero_utm->x() % zero_utm->y() % zero_utm->z()
                             << std::endl;
            }

            std::ofstream ofs(directory + "/special_nodes.csv");
            ofs << "anchor_node " << (anchor_node == nullptr ? -1 : anchor_node->id()) << std::endl;
            ofs << "anchor_edge " << (anchor_edge == nullptr ? -1 : anchor_edge->id()) << std::endl;
            ofs << "floor_node " << (floor_plane_node == nullptr ? -1 : floor_plane_node->id()) << std::endl;

            res.success = true;
            return true;
        }

        /**
         * @brief save map data as pcd
         * @param req
         * @param res
         * @return
         */
        bool save_map_service(hdl_graph_slam::SaveMapRequest &req, hdl_graph_slam::SaveMapResponse &res) {
            return true;
        }

    private:
        // ROS
        ros::NodeHandle nh;
        ros::NodeHandle mt_nh;
        ros::NodeHandle private_nh;
        ros::WallTimer optimization_timer;
        ros::WallTimer map_publish_timer;

        std::unique_ptr <message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
        std::unique_ptr <message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub;
        std::unique_ptr <message_filters::Synchronizer<ApproxSyncPolicy>> sync;

        ros::Subscriber gps_sub;
        ros::Subscriber nmea_sub;
        ros::Subscriber navsat_sub;

        ros::Subscriber imu_sub;
        ros::Subscriber floor_sub;
        ros::Subscriber pose_sub;
        ros::Subscriber command_sub;
        ros::Subscriber dyn_sub;

        ros::Publisher markers_pub;
        ros::Publisher keyframe_pub;
        ros::Publisher dynamic_observation_pub;

        std::string map_frame_id;
        std::string keyframe_frame_id;
        std::string odom_frame_id;
        std::string filtered_points_topic;
        std::string dyn_points_topic;
        std::string agent_keyframe_topic;
        std::string dynamic_object_topic;
        std::string dynamic_observation_topic;
        std::string floor_detection_topic;
        std::string hdl_graph_slam_topic;
        std::string pose_topic;
        std::string agent_no;

        tf::TransformListener tf_listener;

        std::mutex trans_odom2map_mutex;
        Eigen::Matrix4f trans_odom2map;
        ros::Publisher odom2map_pub;
        ros::Publisher dynamic_object_box_pub;


        std::string points_topic;
        std::string agent_position_topic;
        ros::Publisher read_until_pub;
        ros::Publisher map_points_pub;

        ros::ServiceServer dump_service_server;
        ros::ServiceServer save_map_service_server;

        // keyframe queue
        int indexSendKeyframes;
        std::string base_frame_id;
        std::mutex keyframe_queue_mutex;
        std::deque <KeyFrame::Ptr> keyframe_queue;
        std::deque <DynObservation::Ptr> dynamic_object_queue;

        // gps queue
        double gps_time_offset;
        double gps_edge_stddev_xy;
        double gps_edge_stddev_z;
        boost::optional <Eigen::Vector3d> zero_utm;
        std::mutex gps_queue_mutex;
        std::deque <geographic_msgs::GeoPointStampedConstPtr> gps_queue;

        // imu queue
        double imu_time_offset;
        bool enable_imu_orientation;
        double imu_orientation_edge_stddev;
        bool enable_imu_acceleration;
        double imu_acceleration_edge_stddev;
        std::mutex imu_queue_mutex;
        std::deque <sensor_msgs::ImuConstPtr> imu_queue;

        // floor_coeffs queue
        double floor_edge_stddev;
        std::mutex floor_coeffs_queue_mutex;
        std::deque <hdl_graph_slam::FloorCoeffsConstPtr> floor_coeffs_queue;

        // pose queue
        Eigen::Isometry3d initial_pose;
        double initial_yaw;
        Eigen::Isometry3d act_pose;
        Eigen::Quaterniond initial_orientation;
        bool setInitialPosition;
        bool setInitialPosition_orientation;
        bool setInitialPosition_keyframe;

        bool skipping_obs;

        // for map cloud generation
        std::atomic_bool graph_updated;
        double map_cloud_resolution;
        std::mutex keyframes_snapshot_mutex;
        std::vector <KeyFrameSnapshot::Ptr> keyframes_snapshot;
        std::unique_ptr <MapCloudGenerator> map_cloud_generator;

        // graph slam
        // all the below members must be accessed after locking main_thread_mutex
        std::mutex main_thread_mutex;

        int max_keyframes_per_update;
        std::deque <KeyFrame::Ptr> new_keyframes;

        g2o::VertexSE3 *anchor_node;
        g2o::EdgeSE3 *anchor_edge;
        g2o::VertexPlane *floor_plane_node;
        std::vector <KeyFrame::Ptr> keyframes;
        std::unordered_map <ros::Time, KeyFrame::Ptr, RosTimeHash> keyframe_hash;

        std::unique_ptr <GraphSLAM> graph_slam;
        std::unique_ptr <LoopDetector> loop_detector;
        std::unique_ptr <KeyframeUpdater> keyframe_updater;
        std::unique_ptr <NmeaSentenceParser> nmea_parser;

        std::unique_ptr <InformationMatrixCalculator> inf_calclator;

        int keyframe_id;
        int min_points_for_detection;
        int observation_id;
    };


}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::HdlGraphSlamNodelet, nodelet::Nodelet
)
