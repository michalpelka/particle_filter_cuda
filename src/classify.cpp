
#include "particle_filter_fast.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>


int main(int argc, char **argv)
{
	int cudaDevice = 0;
	float normal_vectors_search_radius = 0.2f;
	float curvature_threshold = 0;
	float ground_Z_coordinate_threshold = 1;
	int number_of_points_needed_for_plane_threshold =5;
	int max_number_considered_in_INNER_bucket = 100;
	int max_number_considered_in_OUTER_bucket = 100;

	pcl::console::parse_argument (argc, argv, "-cuda",  cudaDevice);
	pcl::console::parse_argument (argc, argv, "-nsr",   normal_vectors_search_radius);
	pcl::console::parse_argument (argc, argv, "-ct",    curvature_threshold);
	pcl::console::parse_argument (argc, argv, "-z",     ground_Z_coordinate_threshold);
	pcl::console::parse_argument (argc, argv, "-nppt",  number_of_points_needed_for_plane_threshold);
	pcl::console::parse_argument (argc, argv, "-inner", max_number_considered_in_INNER_bucket);
	pcl::console::parse_argument (argc, argv, "-outer", max_number_considered_in_OUTER_bucket);
	
	std::cout << "use case: Classify given poincloud to make it usable in particle filter" << std::endl;
	std::cout << "-cuda : used cuda device                              : " << cudaDevice << std::endl;

	std::cout << "-nsr  : normal_vectors_search_radius                  : " << normal_vectors_search_radius << std::endl;
	std::cout << "-ct   : curvature_threshold                           : " << curvature_threshold << std::endl;
	std::cout << "-z    : ground_Z_coordinate_threshold                 : " << ground_Z_coordinate_threshold << std::endl;
	
	std::cout << "-nppt : number_of_points_needed_for_plane_threshold   : " << number_of_points_needed_for_plane_threshold << std::endl;
	std::cout << "-inner: max_number_considered_in_INNER_bucket  default: " << max_number_considered_in_INNER_bucket << std::endl;
	std::cout << "-outer: max_number_considered_in_OUTER_bucket  default: " << max_number_considered_in_OUTER_bucket << std::endl;

	std::vector<int> ind_pcd;
	

	
	ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");


	if(ind_pcd.size()!=2)
	{
		std::cout << "give input.pcd output.pcd" << std::endl;
		return -1;
	}


	std::string inputCloudFn (argv[ind_pcd[0]]);
	std::string outputCloudFn (argv[ind_pcd[1]]);


	std::cout << "input pointcloud  : " << inputCloudFn << std::endl;
	std::cout << "output pointcloud : " << outputCloudFn << std::endl;

	pcl::PointCloud<pcl::PointXYZ> input;
	pcl::PointCloud<Semantic::PointXYZL> output;
	pcl::io::loadPCDFile(inputCloudFn, input);

	CPointcloudClassifier classifier(cudaDevice,
			 normal_vectors_search_radius,
			 curvature_threshold,
			 ground_Z_coordinate_threshold,
			 number_of_points_needed_for_plane_threshold,
			 max_number_considered_in_INNER_bucket,
			 max_number_considered_in_OUTER_bucket);

	classifier.classify(input, output);
	std::cout << "saving \n";
	pcl::io::savePCDFile(outputCloudFn, output);

}
