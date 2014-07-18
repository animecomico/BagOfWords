/*
 * =====================================================================================
 *
 *       Filename:  bow_try1.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/14/2014 02:09:13 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

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


namespace fs = boost::filesystem;
int minHess = 400;
cv::SurfFeatureDetector surf_detector(minHess);
cv::SurfDescriptorExtractor surf_extractor;
cv::Mat surf_descriptors;
int max_nodescriptors = 0;
int abs_no_files = 0;
int no_classes = 0;

//for enconding images with BOW
cv::Mat bowDescriptor;
cv::Ptr<cv::DescriptorMatcher> dmatcher = cv::DescriptorMatcher::create("FlannBased");	
cv::Ptr<cv::DescriptorExtractor> dextractor = cv::DescriptorExtractor::create("SURF");
cv::BOWImgDescriptorExtractor bowDE(dextractor,dmatcher);


void process_dir(const fs::path& basepath)
{
	int no_files = 0;
	for(fs::directory_iterator iter(basepath), end; iter != end; ++iter)
	{
		fs::directory_entry entry = *iter;
		if(fs::is_directory(entry.path()))
		{
			std::cout << "Processing directory: " << entry.path().string() << std::endl;
			process_dir(entry.path());
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
					std::vector<cv::KeyPoint> surf_keypoints;
					surf_detector.detect(img,surf_keypoints);
					if(surf_keypoints.empty())
					{
						std::cerr << "Could not find points in image: " << entryPath.string();
						std::cerr << std::endl;
					}
					else
					{

						cv::Mat local_descriptors;
						surf_extractor.compute(img,surf_keypoints,local_descriptors);
						//bowTrainer.add(local_descriptors);
						surf_descriptors.push_back(local_descriptors);
						surf_keypoints.clear();
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

int class_test = 0;
void process_dir2(const fs::path& basepath, cv::Mat& descriptors, cv::Mat& labels)
{
	int no_files = 0;
	for(fs::directory_iterator iter(basepath), end; iter != end; ++iter)
	{
		fs::directory_entry entry = *iter;
		if(fs::is_directory(entry.path()))
		{
			std::cout << "Processing directory: " << entry.path().string() << std::endl;
			class_test++;
			process_dir2(entry.path(),descriptors,labels);
		}
		else
		{
			fs::path entryPath = entry.path();
			if(entryPath.extension()==".jpg")
			{
			//	std::cout << "Processing file: " << entry.path().string() << std::endl;
				no_files++;
				cv::Mat img = cv::imread(entryPath.string(),CV_LOAD_IMAGE_COLOR);
				if(!img.empty())
				{
					std::vector<cv::KeyPoint> surf_keypoints;
					surf_detector.detect(img,surf_keypoints);
					if(surf_keypoints.empty())
					{
						std::cerr << "Could not find points in image: " << entryPath.string();
						std::cerr << std::endl;
					}
					else
					{

						cv::Mat bowDescriptor;
						bowDE.compute(img,surf_keypoints,bowDescriptor);
						descriptors.push_back(bowDescriptor);
						labels.push_back(float(class_test));
						surf_keypoints.clear();

						//std::cout << local_descriptors.rows << std::endl;	
						~bowDescriptor;

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
	std::string file_path = "training_sets/flavia_leaves_c";
	process_dir(fs::path(file_path));

	std::cout << "Total no of descriptors: "<<surf_descriptors.size() << std::endl;
	std::cout << "Max no of descriptors in an image: " << max_nodescriptors << std::endl;
	t = clock()-t;
	std::cout << "Descriptors processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//
	
	//--- SAVE DESCRIPTORS INTO A FILE ---//
	cv::FileStorage fstore("test.yml", cv::FileStorage::WRITE);
	fstore << "noDocuments" << abs_no_files;
	fstore << "noClasses" << no_classes;	
	fstore << "totalDescriptors" << surf_descriptors.rows;
	fstore << "maxDescriptors" << max_nodescriptors;
	fstore << "procTime" << ((float)t)/CLOCKS_PER_SEC;
	fstore << "matDescriptors" << surf_descriptors;
	fstore.release();
	//--- ---//
	//--- READ DESCRIPTORS FROM FILE ---//
	std::cout << "***" << std::endl;
	//std::cout << surf_descriptors.row(1) << std::endl;
	cv::FileStorage fstore2("test.yml", cv::FileStorage::READ);
	std::cout << (int)fstore2["noDocuments"] << std::endl;
	std::cout << (int)fstore2["noClasses"] << std::endl;
	std::cout << (int)fstore2["totalDescriptors"] << std::endl;
	std::cout << (int)fstore2["maxDescriptors"] << std::endl;
	std::cout << (float)fstore2["procTime"] << std::endl;
	/*
	cv::Mat tmp;
	fstore2["matDescriptors"] >> tmp;
	std::cout << tmp.row(1) << std::endl;*/
	fstore2.release();
	

	//--- BUILD A DICTIONARY ---//
	//int dictionary_size = 1000;
	cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int cluster_attempts = 1;
	int dictionary_size = atoi(argv[1]);
	std::cout << "Dictionary size: " << dictionary_size << std::endl; 
	cv::BOWKMeansTrainer bowTrainer(dictionary_size,tc,cluster_attempts, cv::KMEANS_PP_CENTERS);

	bowTrainer.add(surf_descriptors);
	t = clock();
	cv::Mat my_dictionary = bowTrainer.cluster();
	t = clock()-t;
	std::cout << "Dictionary processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	//--- ---//

	//--- Naive Bayes for classification ---//
	cv::NormalBayesClassifier nb_classifier;
	cv::Mat training_data(0,dictionary_size,CV_32FC1);
	cv::Mat labels(0,1,CV_32FC1);
	cv::Mat labels2(0,1,CV_32FC1);
	cv::Mat eval_data(0,dictionary_size,CV_32FC1);
	cv::Mat results;

	bowDE.setVocabulary(my_dictionary);
	process_dir2(fs::path(file_path),training_data,labels);

	std::cout << "Training classifier..." << std::endl;
	std::cout << training_data.size() << " * " << labels.size() << std::endl;
	if(training_data.type() == CV_32FC1)
	{std::cout << "yes1" << std::endl;}
	if(labels.type() == CV_32FC1)
	{std::cout << "yes2" << std::endl;}

	t = clock();
	nb_classifier.train(training_data,labels,cv::Mat(),cv::Mat(),false);
	t = clock() - t;
	std::cout << " Training processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	//--- ---//

	//--- Evaluation images BOW encoding ---//

	std::string file_path2 = "test_sets/flavia_leaves_c";
	process_dir2(fs::path(file_path2),eval_data,labels2);

	std::cout << "Evaluating classifier..." << std::endl;
	t = clock();
	nb_classifier.predict(eval_data,&results);	
	t = clock()-t;
	std::cout << " Classifier evaluation time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	std::cout << "Classifier Results" << std::endl;
	std::cout << results << std::endl;
	
	

	return 0;
}				/* ----------  end of function main  ---------- */


