/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

using namespace cv;//

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame 
        extractKeyPoints();
        computeDescriptors();
        // compute the 3d position of features in ref frame 
        setRef3DPoints();
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
	setCurr3DPoints();//
        //poseEstimationPnP();
	poseEstimationICP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            cout << "ref_->T_c_w_: " << endl << ref_->T_c_w_ << endl;//
	    curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
	    cout << "curr_->T_c_w_: " << endl << curr_->T_c_w_ << endl;//
	    
            ref_ = curr_;
            setRef3DPoints();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
		//cv::waitKey( 0 );
            }
            cv::waitKey( 0 );
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    orb_->detect ( curr_->color_, keypoints_curr_ );
}

void VisualOdometry::computeDescriptors()
{
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::featureMatching()
{
    // match desp_ref and desp_curr, use OpenCV's brute force match 
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher ( cv::NORM_HAMMING );
    matcher.match ( descriptors_ref_, descriptors_curr_, matches );
    cout << "matches: " << matches.size() << endl;//
    
/*    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches( ref_->color_, keypoints_ref_, curr_->color_, keypoints_curr_, matches, imgMatches );
    cv::imshow ( " matches " , imgMatches );
    cv::imwrite( "./data/matches.png " , imgMatches );
    cv::waitKey( 0 ); */
    
    /*//方法1 select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    feature_matches_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            feature_matches_.push_back(m);
        }
    }*/
    
    //方法2 select the best matches
    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_ref_.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    feature_matches_.clear();
    for ( int i = 0; i < descriptors_ref_.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
	//if ( matches[i].distance <= (2*min_dist) )
        {
            feature_matches_.push_back ( matches[i] );
        }
    }
    
    cout<<"good matches: "<<feature_matches_.size()<<endl;
/*    //显示good matches  
    cv::Mat imgMatches;
    cv::drawMatches( ref_->color_, keypoints_ref_, curr_->color_, keypoints_curr_, feature_matches_, imgMatches );
    cv::imshow( " good matches " , imgMatches );
    cv::imwrite( "./data/good_matches.png " , imgMatches );
    cv::waitKey( 0 );   */ 

    ransac(feature_matches_, keypoints_ref_,keypoints_curr_);
}

//RANSAC算法实现 下面成员函数要有命名空间VisualOdometry::！！！
void VisualOdometry::ransac(vector<DMatch> feature_matches_,vector<KeyPoint> keypoints_ref_,vector<KeyPoint> keypoints_curr_)
{
    //定义保存匹配点对坐标  
    vector<cv::Point2f> srcPoints(feature_matches_.size()),dstPoints(feature_matches_.size());  
    //保存从关键点中提取到的匹配点对的坐标  
    for ( int  i=0;i<feature_matches_.size();i++)  
    {  
        srcPoints[i]=keypoints_ref_[feature_matches_[i].queryIdx].pt;  
        dstPoints[i]=keypoints_curr_[feature_matches_[i].trainIdx].pt;
	/*//有的例程是下面的：
	srcPoints[i]=keypoints_ref_[feature_matches_[i].trainIdx].pt;  
        dstPoints[i]=keypoints_curr_[feature_matches_[i].queryIdx].pt;*/
    }  
    //保存计算的单应性矩阵  
    Mat homography;  
    //保存点对是否保留的标志  
    vector<unsigned  char > inliersMask(srcPoints.size());   
    //匹配点对进行RANSAC过滤  
    //RANSAC algorithm与下一行等价
    //homography = findHomography(srcPoints,dstPoints,CV_FM_RANSAC, 5.0, inliersMask);
    homography = findHomography(srcPoints,dstPoints,RANSAC,50,inliersMask,2000,0.995); 
    //cout << "homography:" << endl << homography << endl;
    //homography = findHomography(srcPoints,dstPoints,LMEDS,5,inliersMask); //least-median algorithm
    //homography = findHomography(srcPoints,dstPoints,LMEDS,5,inliersMask,2000,0.995);
    //RANSAC过滤后的点对匹配信息  
    //vector<DMatch> matches_ransac;//在头文件中有声明，不能再次申明了！！！
    //手动的保留RANSAC过滤后的匹配点对  
    
    num_inliersMask = inliersMask.size();
    cout <<"num_inliersMask: "<<num_inliersMask<<endl;
    matches_ransac.clear();
    for ( int  i=0;i<inliersMask.size();i++)  
    {  
        if (inliersMask[i])  
        {  
            matches_ransac.push_back(feature_matches_[i]);  
            //cout<<"第"<<i<<"对匹配："<<endl;  
            //cout<<"queryIdx:"<<matches[i].queryIdx<<"\ttrainIdx:"<<matches[i].trainIdx<<endl;  
            //cout<<"imgIdx:"<<matches[i].imgIdx<<"\tdistance:"<<matches[i].distance<<endl;  
        }  
    } 

    //返回RANSAC过滤后的点对匹配信息  
    //return  matches_ransac;  
    cout <<"matches_ransac: "<<matches_ransac.size()<<endl;
    num_inliers_ = matches_ransac.size();//
    cout <<"num_inliers_: "<<num_inliers_<<endl;
    
    
    //显示matches_ransac 
    cv::Mat imgMatches;
    cv::drawMatches( ref_->color_, keypoints_ref_, curr_->color_, keypoints_curr_,matches_ransac, imgMatches );
    cv::imshow( " matches_ransac " , imgMatches );
    cv::imwrite( "./data/matches_ransac.png " , imgMatches );
    //cv::waitKey( 0 );
}

void VisualOdometry::setRef3DPoints()
{
    // select the features with depth measurements 
    pts_3d_ref_.clear();
    descriptors_ref_ = Mat();
    keypoints_ref_ = keypoints_curr_;//
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {
        //方法１
        //double d = ref_->findDepth(keypoints_curr_[i]);
	//方法２
	double d = ref_->depth_.ptr<unsigned short> ( int ( keypoints_curr_[i].pt.y ) ) [ int ( keypoints_curr_[i].pt.x ) ];
	d = d/5000;
	
	//cout << "d:" << d << " d_:" << d_ << endl;//
	
        if ( d > 0)
        {
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
            );
            pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
            descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
}

void VisualOdometry::setCurr3DPoints()
{
    // select the features with depth measurements 
    pts_3d_curr_.clear();
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {           
        //方法１
        //double d_ = curr_->findDepth(keypoints_curr_[i]);
	//方法２
	double d = curr_->depth_.ptr<unsigned short> ( int ( keypoints_curr_[i].pt.y ) ) [ int ( keypoints_curr_[i].pt.x ) ];
	d = d/5000;

        if ( d > 0)
        {
            Vector3d p_cam = curr_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
            );
	    pts_3d_curr_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
        }
    }
}

//编写ICP，替换PnP实现vo
void VisualOdometry::poseEstimationICP()
{
    Mat R, t, R_inv, t_inv, rvec; //ziji tianjia
    vector<cv::Point3f> pts3d1;
    vector<cv::Point3f> pts3d2;
    // select the features with depth measurements 
   
    for ( DMatch m:matches_ransac ) //使用findhomography再次筛选的结果 使用LMEDS方法，未使用RANSAC algorithm。
//    for ( DMatch m:feature_matches_ )   //直接使用这个的话，在运行的窗口中根本看不到相机的位姿？？？？？？？？？
    {
        pts3d1.push_back( pts_3d_ref_[m.queryIdx] );
        pts3d2.push_back( pts_3d_curr_[m.trainIdx] );
        
    }
    
    //把第二帧到第一帧的变换转化第一帧到第二帧的变换 法１
    vector<cv::Point3f> pts3dtemp;
    pts3dtemp.clear();
    pts3dtemp = pts3d1;
    pts3d1.clear();
    pts3d1 = pts3d2;
    pts3d2.clear();
    pts3d2 = pts3dtemp;
    
    //SVD方法：
    cv::Point3f p1, p2;     // center of mass
    int N = pts3d1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts3d1[i];
        p2 += pts3d2[i];
    }
    p1 /=  N;
    p2 /=  N;
    vector<cv::Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts3d1[i] - p1;
        q2[i] = pts3d2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    //cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
/*https://github.com/gaoxiang12/slambook/issues/106    ???
    //利用SVD求解3D-3D变换，需要U和V的行列式同号。换言之，旋转矩阵的行列式只能为1，不能为-1。
    if (U.determinant() * V.determinant() < 0)
    {
         for (int x = 0; x < 3; ++x)
         {
             U(x, 2) *= -1;
         }
    }*/
    
    //cout<<"U="<<U<<endl;
    //cout<<"V="<<V<<endl;
    
    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
    
    //Eigen::Vector3d rvec = R.log();
    //rvec = rvec.transpose();
    
    /*//把第二帧到第一帧的变换转化第一帧到第二帧的变换 法２
    R_inv = R.t();//
    t_inv = -R.t()*t; */
    
    R_inv = R;
    t_inv = t; //和上面把第二帧到第一帧的变换转化第一帧到第二帧的变换 法１－配合使用
    
    //cout << "R：" << endl << R_inv << endl;
        
    cv::Rodrigues(R_inv, rvec);
    
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( t_inv.at<double>(0,0), t_inv.at<double>(1,0), t_inv.at<double>(2,0))
    );
      
    cout << "pose-T_c_r_estimated_：" << endl << T_c_r_estimated_ << endl;
    //cv::waitKey(0);
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    cout<<"motion的模: "<<d.norm()<<endl;//
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    
    cout << "rot.norm(): " << rot.norm() << endl;
    cout << "trans.norm(): " << trans.norm() << endl;
    
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

}