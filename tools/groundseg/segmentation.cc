#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>

#include "ground_segmentation.h"

namespace py = pybind11;

std::vector<int> segment(const Eigen::MatrixXf & points,
                         const bool visualize,
                         const double max_dist_to_line,
                         const double max_slope,
                         const double long_threshold,
                         const double max_long_height,
                         const double max_start_height,
                         const double sensor_height,
                         const double line_search_angle,
                         const double r_min, 
                         const double r_max, 
                         const double max_fit_error,
                         const int n_threads){
    // create a pcl point cloud with the input nx3 matrix 
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.resize(points.rows());

    // assign coordinates
    for (int i=0;  i<points.rows(); ++i) {
        Eigen::Vector3f point = points.row(i);
        cloud.points[i].getVector3fMap() = point;
    }

    //
    GroundSegmentationParams params;

    // params.visualize = false;    // # visualize segmentation result - USE ONLY FOR DEBUGGING
    params.visualize = visualize;    // # visualize segmentation result - USE ONLY FOR DEBUGGING
    params.n_bins = 120;        // # number of radial bins
    params.n_segments = 360;    // # number of radial segments.
    
    // params.max_dist_to_line = 0.1; // # maximum vertical distance of point to line to be considered ground.
    // params.max_slope = 0.4;         // # maximum slope of a ground line.
    // params.long_threshold = 2.0;    // # distance between points after which they are considered far from each other.
    // params.max_long_height = 0.2;   // # maximum height change to previous point in long line.
    // params.max_start_height = 0.4;  // # maximum difference to estimated ground height to start a new line.
    // params.sensor_height = 1.84;     // # sensor height above ground.
    // params.line_search_angle = 0.2; // # how far to search in angular direction to find a line [rad].

    params.max_dist_to_line = max_dist_to_line; // # maximum vertical distance of point to line to be considered ground.
    params.max_slope = max_slope;         // # maximum slope of a ground line.
    params.long_threshold = long_threshold;    // # distance between points after which they are considered far from each other.
    params.max_long_height = max_long_height;   // # maximum height change to previous point in long line.
    params.max_start_height = max_start_height;  // # maximum difference to estimated ground height to start a new line.
    params.sensor_height = sensor_height;     // # sensor height above ground.
    params.line_search_angle = line_search_angle; // # how far to search in angular direction to find a line [rad].

    params.n_threads = n_threads;           // # number of threads to use.

    // double r_min = 0.8;         // # minimum point distance.
    // double r_max = 70.4;          // # maximum point distance.
    // double max_fit_error = 0.1; // # maximum error of a point during line fit.

    params.r_min_square = r_min*r_min;
    params.r_max_square = r_max*r_max;
    params.max_error_square = max_fit_error * max_fit_error;
    
    GroundSegmentation segmenter(params);
    std::vector<int> labels;
    segmenter.segment(cloud, &labels);

    return labels;
}

PYBIND11_MODULE(segmentation, m) {
    m.doc() = "LiDAR ground segmentation";
    
    m.def("segment", &segment, "ground segmentation", 
          py::arg("points"), 
          py::arg("visualize")=false, 
          py::arg("max_dist_to_line")=0.1,
          py::arg("max_slope")=0.4,
          py::arg("long_threshold")=2.0,
          py::arg("max_long_height")=0.2, 
          py::arg("max_start_height")=0.4,
          py::arg("sensor_height")=1.84,
          py::arg("line_search_angle")=0.2,
          py::arg("r_min")=0.8,
          py::arg("r_max")=70.4,
          py::arg("max_fit_error")=0.1,
          py::arg("n_threads")=1);
}