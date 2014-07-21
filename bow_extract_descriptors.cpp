/* =============================================================*/
/* ---           BOW EXTRACT DESCRIPTORS - SOURCE FILE       ---*/
/* FILENAME: bow_extract_descriptors.cpp 
 *
 * DESCRIPTION: source file to compute descriptors of a data base
 * and save it into a YAML file. This module is done separetely
 * with the purpose of using it as a reference to set the
 * parameters of the BOW algorithm later on
 *
 * VERSION: 1.0
 *
 * CREATED: 07/15/2014
 *
 * COMPILER: g++
 *
 * AUTHOR: ARTURO GOMEZ CHAVEZ
 * ALIAS: BOSSLEGEND33
 * 
 * ============================================================ */

//c++ libraries
#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include "boost/filesystem.hpp"
//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
//Defines
#define DATABASE_PATH "training_sets/flavia_leaves_b"
#define DESCRIP_MAT_NAME "descrip_train_mat.yml"
//Namespaces
namespace fs = boost::filesystem;

//--- GLOBAL VARIABLES ---//
int minHess = 400;
cv::SurfFeatureDetector feature_detector(minHess);
cv::SurfDescriptorExtractor feature_extractor;
cv::Mat feature_descriptors;
int max_nodescriptors = 0;
int abs_no_files = 0;
int no_classes = 0;
//--- ---//

void extract_descriptors(const fs::path& basepath)
{
	int no_files = 0;
	for(fs::directory_iterator iter(basepath), end; iter != end; ++iter)
	{
		fs::directory_entry entry = *iter;
		if(fs::is_directory(entry.path()))
		{
			std::cout << "Processing directory: " << entry.path().string() << std::endl;
			extract_descriptors(entry.path());
			no_classes++;
		}
		else
		{
			fs::path entryPath = entry.path();
			if(entryPath.extension()==".jpg")
			{
			//	std::cout << "Processing file: " << entry.path().string() << std::endl;
				no_files++;
				abs_no_files++;
				cv::Mat img = cv::imread(entryPath.string(),CV_LOAD_IMAGE_COLOR);
				if(!img.empty())
				{
					std::vector<cv::KeyPoint> feature_keypoints;
					feature_detector.detect(img,feature_keypoints);
					if(feature_keypoints.empty())
					{
						std::cerr << "Could not find points in image: " << entryPath.string();
						std::cerr << std::endl;
					}
					else
					{

						cv::Mat local_descriptors;
						feature_extractor.compute(img,feature_keypoints,local_descriptors);
						feature_descriptors.push_back(local_descriptors);
						feature_keypoints.clear();
						if(local_descriptors.rows > max_nodescriptors)
							max_nodescriptors = local_descriptors.rows;
						//std::cout << local_descriptors.rows << std::endl;	
						~local_descriptors;

					}
				}
				else
				{
					std::cerr << "Could not read image: " << entryPath.string() << std::endl;
				}
												
			}												
		}							
		
	}		
	std::cout << "No files: " << no_files << std::endl;

}



int main ( int argc, char *argv[] )
{
	
	//---  VARIABLES ---//
	//for measuring processing time
	clock_t t;	
	// --- ---//
	
	//--- DESCRIPTORS EXTRACTION ---//
	t = clock();
	std::string file_path = DATABASE_PATH;
	extract_descriptors(fs::path(file_path));

	std::cout << "Database path: " << file_path << std::endl;
	std::cout << "Total no of descriptors: "<< feature_descriptors.size() << std::endl;
	std::cout << "Max no of descriptors in an image: " << max_nodescriptors << std::endl;
	t = clock()-t;
	std::cout << "Descriptors processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//
	
	//--- SAVE DESCRIPTORS INTO A FILE ---//
	cv::FileStorage fstore(DESCRIP_MAT_NAME, cv::FileStorage::WRITE);
	fstore << "database" << file_path;
	fstore << "noDocuments" << abs_no_files;
	fstore << "noClasses" << no_classes;	
	fstore << "totalDescriptors" << feature_descriptors.rows;
	fstore << "maxDescriptors" << max_nodescriptors;
	fstore << "procTime" << ((float)t)/CLOCKS_PER_SEC;
	fstore << "matDescriptors" << feature_descriptors;
	fstore.release();
	//--- ---//

	/*
	//+++ FOR DEBUGGING +++//
	//--- READ DESCRIPTORS FROM FILE ---//
	std::cout << "***" << std::endl;
	std::cout << feature_descriptors.row(1) << std::endl;
	cv::FileStorage fstore2("test.yml", cv::FileStorage::READ);
	std::cout << (int)fstore2["noDocuments"] << std::endl;
	std::cout << (int)fstore2["noClasses"] << std::endl;
	std::cout << (int)fstore2["totalDescriptors"] << std::endl;
	std::cout << (int)fstore2["maxDescriptors"] << std::endl;
	std::cout << (float)fstore2["procTime"] << std::endl;
	cv::Mat tmp;
	fstore2["matDescriptors"] >> tmp;
	std::cout << tmp.row(1) << std::endl;
	fstore2.release();
	*/ 
	

	return 0;
}				/* ----------  end of function main  ---------- */


