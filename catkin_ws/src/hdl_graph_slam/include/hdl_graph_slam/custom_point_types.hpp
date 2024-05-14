//
// Created by greve on 1/10/23.
//

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#ifndef CATKIN_WS_POINT_TYPES_H
#define CATKIN_WS_POINT_TYPES_H

#define PCL_ADD_UNION_POINT4D \
  union EIGEN_ALIGN16 { \
    float data[4]; \
    struct { \
      float x; \
      float y; \
      float z; \
    }; \
  };


#define PCL_ADD_POINT4D \
  PCL_ADD_UNION_POINT4D \
  PCL_ADD_EIGEN_MAPS_POINT4D

#define PCL_ADD_RGB \
  PCL_ADD_UNION_RGB \
  PCL_ADD_EIGEN_MAPS_RGB

namespace hdl_graph_slam {

    struct EIGEN_ALIGN16 POINT_XYZ_INTENSITY_CLASS_INSTANCE
            {
                    PCL_ADD_POINT4D;
                    float intensity;
                    float class_label;
                    float instance;
                    PCL_MAKE_ALIGNED_OPERATOR_NEW
            };


}

POINT_CLOUD_REGISTER_POINT_STRUCT (hdl_graph_slam::POINT_XYZ_INTENSITY_CLASS_INSTANCE,
(float, x, x)
(float, y, y)
(float, z, z)
(float, intensity, intensity)
(float, class_label, class_label)
(float, instance, instance)
)
#endif //CATKIN_WS_POINT_TYPES_H
