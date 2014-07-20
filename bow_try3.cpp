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
#include <string>
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
#define TRAIN_PATH "training_sets/flavia_leaves_b"
#define TEST_PATH "test_sets/flavia_leaves_b"
#define DESCRIP_MAT_NAME "des_train_mat.yml"

namespace fs = boost::filesystem;
int minHess = 400;
cv::SurfFeatureDetector surf_detector(minHess);
cv::SurfDescriptorExtractor surf_extractor;
cv::Mat training_descriptors;
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
						training_descriptors.push_back(local_descriptors);
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
	std::string class_name = basepath.string();
	class_name.erase(class_name.begin(),class_name.end()-1);

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
						//std::cout << class_name.c_str() << std::endl;
						labels.push_back( float( atoi(class_name.c_str()) ) );
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
	
	std::cout << "+++ BOW FOR DATA SET +++" << std::endl;
	std::cout << TRAIN_PATH << std::endl;

	//--- DESCRIPTORS EXTRACTION ---//
	std::string file_path = TRAIN_PATH;
	//read descriptors from file
	std::cout << "*** TRAIN DESCRIPTORS INFO ***" << std::endl;
	cv::FileStorage fstore_descrip(DESCRIP_MAT_NAME, cv::FileStorage::READ);
	std::cout << "No Documents: " << (int)fstore_descrip["noDocuments"] << std::endl;
	std::cout << "No Classes: " << (int)fstore_descrip["noClasses"] << std::endl;
	std::cout << "No total descriptors: " << (int)fstore_descrip["totalDescriptors"] << std::endl;
	std::cout << "No max desrcriptors in an image: " << (int)fstore_descrip["maxDescriptors"] << std::endl;
	std::cout << "Descriptors processing time" << (float)fstore_descrip["procTime"] << std::endl;
	std::cout << std::endl;
	fstore_descrip["matDescriptors"] >> training_descriptors;
	fstore_descrip.release();
	

	//--- BUILD A DICTIONARY ---//
	cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int cluster_attempts = 1;
	int dictionary_size = atoi(argv[1]);
	std::cout << "*** BOW DICTIONARY INFO ***" << std::endl;
	std::cout << "Dictionary size: " << dictionary_size << std::endl; 
	cv::BOWKMeansTrainer bowTrainer(dictionary_size,tc,cluster_attempts, cv::KMEANS_PP_CENTERS);

	bowTrainer.add(training_descriptors);
	t = clock();
	cv::Mat my_dictionary = bowTrainer.cluster();
	t = clock()-t;
	std::cout << "Dictionary processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << std::endl;
	//--- ---//

	//--- NAIVE BAYES FOR CLASSIFICATION ---//
	cv::NormalBayesClassifier nb_classifier;
	cv::Mat training_data(0,dictionary_size,CV_32FC1);
	cv::Mat labels(0,1,CV_32FC1);

	//set the dictionary to bow descriptor extractor
	std::cout << "*** CLASSIFIER TRAINING ***" << std::endl;
	bowDE.setVocabulary(my_dictionary);
	process_dir2(fs::path(file_path),training_data,labels);

	// +++ for debugging - can  be commented +++//
	std::cout << training_data.size() << " * " << labels.size() << std::endl;
	if(training_data.type() == CV_32FC1)
	{std::cout << "training data matrix accepted" << std::endl;}
	if(labels.type() == CV_32FC1)
	{std::cout << "labels matrix accepted" << std::endl;}
	// +++ +++ //

	t = clock();
	nb_classifier.train(training_data,labels,cv::Mat(),cv::Mat(),false);
	t = clock() - t;
	nb_classifier.save("nbModel_flavia_leaves_b.yml","nbModel_flavia_leaves_b");
	std::cout << " Training processing time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
	std::cout << std::endl;

	//--- ---//

	//--- BOW ENCODING OF TEST SET AND EVALUATION ---//
	cv::Mat ground_truth(0,1,CV_32FC1);
	cv::Mat eval_data(0,dictionary_size,CV_32FC1);
	cv::Mat results;
	std::string file_path2 = TEST_PATH;
	process_dir2(fs::path(file_path2),eval_data,ground_truth);
	double accuRate = 0.;

	std::cout << "*** CLASSIFIER EVALUATION ***" << std::endl;
	t = clock();
	nb_classifier.predict(eval_data,&results);	
	t = clock()-t;
	std::cout << " Classifier evaluation time:" << std::endl;
	std::cout << t << " clicks " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;

	accuRate = 1. -( (double) cv::countNonZero(ground_truth - results) / eval_data.rows);
	std::cout << "Accuracy rate: " << accuRate << std::endl;
	std::cout << "Classifier Results" << std::endl;
	std::cout << results << std::endl << std::endl;
	
	

	return 0;
}				/* ----------  end of function main  ---------- */


