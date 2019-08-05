#include <opencv2/opencv.hpp>
#include <iostream>
#include "features2d.hpp"

#define CV_MAT_ELEM2(src,dtype,y,x) \
        (dtype*)(src.data+src.step[0]*y+src.step[1]*x)


using namespace std;
using namespace cv;

void videotrac();
void pcaprocess();
void ave(int num);
void findstraight(int num);
void eucluster(int num, vector<Point> points, vector<vector<Point>>& clusters, int thre,int minnum,bool save);
void cluster(vector<Point> points, vector<vector<Point>>& clusters, float thre,int minnum);
void setcolor();
Mat polyfit(std::vector<Point> &chain, int n);
void myfit(vector<vector<Point>>& clusters,int n);

vector<Vec3i> color;
vector<Point> points;
vector<vector<Point>> clusters;
vector<Point> curve1;

int main()
{   
	setcolor();
	findstraight(1885);
	eucluster(1885,points,clusters,20,100,1);
	return 0;
}

//设置聚类的显示颜色
void setcolor() {
	color.push_back(Vec3i(255, 0, 0));
	color.push_back(Vec3i(0, 255, 0));
	color.push_back(Vec3i(0, 0, 255));
	color.push_back(Vec3i(0, 100, 255));
	color.push_back(Vec3i(100, 0, 255));
	color.push_back(Vec3i(100, 190, 255));
}
//提取每帧的目标，使用opencv的MOG提取接口，可以试着使用其他背景提取方法
void videotrac() {
	Mat frame;
	Mat frame_grey;
	Mat bgImg;
	Mat fgMaskMOG2;

	std::vector<cv::Point2f> corners;
	vector<vector<Point>>contours;
	vector<Point> trac;

	Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(100, 36.0, true);

	string videoFilename = "5.mp4";
	VideoCapture capture(videoFilename);

	int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int numframe = 0;
	Mat maintra(height, width, CV_8UC1, cv::Scalar::all(0));


	if (!capture.isOpened())
	{
		cerr << "Unable to open video file: " << videoFilename << endl;
		return ;
	}


	while (true) {
		int key = waitKey(1);
		if (key == 'q' || key == 27)
			return ;
		if (key == 'p') {
			key = waitKey();
		}
		if (key =='c') {
			string filename = "E:\\track\\track\\trajectory\\_save_" + to_string(numframe) + ".jpg";
	     	imwrite(filename, maintra);
			maintra.setTo(Scalar(0));
		}
		if (!capture.read(frame))
			break;

		numframe++;
		cout << numframe << endl;
		//前景提取，结果为fgMaskMOG2
		cv::cvtColor(frame, frame_grey, cv::COLOR_BGR2GRAY);

		pMOG2->apply(frame, fgMaskMOG2);
		pMOG2->getBackgroundImage(bgImg);
		
		
		imshow("origin", fgMaskMOG2);

		threshold(fgMaskMOG2, fgMaskMOG2, 200, 255, CV_THRESH_BINARY);
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(10, 10)));//fill black holes
		morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3))); //fill white holes
	
		medianBlur(fgMaskMOG2, fgMaskMOG2, 5);
		imshow("for2sad2", fgMaskMOG2);

		


		findContours(fgMaskMOG2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		frame_grey = cv::Scalar::all(0);

		for (int i = 0; i < contours.size(); i++)

		{
			Rect rct;
			if (contourArea(contours[i]) > 600 && contourArea(contours[i]) < 10000) {
				rct = boundingRect(contours[i]);
				if (rct.y >height / 4 - rct.height / 2) {
					rectangle(frame, rct, Scalar(0, 255, 0), 2);
					Point cen; cen.x = rct.x + rct.width / 2; cen.y = rct.y + rct.height / 2;
					trac.push_back(cen);
					((uchar*)maintra.data)[width*cen.y + cen.x] += 255;
					circle(frame, cen, 4, cv::Scalar(0, 0, 255));
				}
			}
		}

		cv::drawContours(frame_grey, contours, -1, cv::Scalar::all(255));
		imshow("forc", frame_grey);

		
		for (int i = 0; i < trac.size(); i++) {
			//cout << trac.size() << endl;
			//circle(frame, trac[i], 2, cv::Scalar(255, 0, 0));
		}


		imshow("maintra", maintra);
		imshow("for", fgMaskMOG2);
		imshow("original", frame);
		imshow("mask", bgImg);
		if (numframe % 200 == 0) {
			cout << "in" << endl;
			string filename = "E:\\track\\track\\trajectory\\" + to_string(numframe) + ".jpg";
			imwrite(filename, frame);
			filename = "E:\\track\\track\\trajectory\\binary_" + to_string(numframe) + ".jpg";
			imwrite(filename, maintra);
		}

	}
	return ;
}
//此函数用pca提取子块的主要方向，实际实验中并不起作用，所以没有采用
void pcaprocess() {
	Mat trac;
	struct point {
		int x;
		int y;
	};
	vector<point>  tracpoint;
	trac=imread("E:\\track\\track\\trajectory\\binary_400.jpg");

	int offsetx;
	int offsety;
	int step = 100;
	int width = trac.cols;
	int height = trac.rows;

	Mat dir(height, width, CV_8UC1,cv::Scalar::all(0));

	for (offsety = 0; offsety <height-step; offsety += step) {
		for (offsetx = 0; offsetx < width - step; offsetx += step) {
			tracpoint.clear();
			for(int i=0;i<step;i++)
				for (int j = 0; j < step; j++) {
					if (((uchar*)trac.data)[(offsety + i)*width + offsetx + j]) {
						point tem;
						tem.x = offsetx + j; tem.y = offsety + i;
						tracpoint.push_back(tem);
					}
				}
			if (tracpoint.size()) {
				Mat sample(2, tracpoint.size(), CV_16UC1);
				for (int i = 0; i < tracpoint.size(); i++) {
					((ushort*)trac.data)[i] = tracpoint[i].x;
					((ushort*)trac.data)[i + tracpoint.size()] = tracpoint[i].y;
				}
				PCA pca(sample, noArray(), 1,1);
				//cout << pca.eigenvectors<<endl;
				//visualze
				float sx = ((float*)pca.eigenvectors.data)[0];
				float sy = ((float*)pca.eigenvectors.data)[1];
				int endx = step * sx / 2;
				int endy = step * sy / 2;
				line(trac, Point(offsetx+step/2, offsety + step / 2), Point(offsetx+endx, offsety + endy), Scalar(255), 5);
				
			}
		
		}
		imshow("dir", trac);
	}
	waitKey();
}
//提取直线，num为提取直线的图片号
void findstraight(int num) {
	Mat tra;
	string filename = "E:\\track\\track\\trajectory\\_save_" + to_string(num) + ".jpg";
	tra = imread(filename, 0);
	threshold(tra, tra, 100, 255, 0);

	int width = tra.cols;
	int height = tra.rows;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (((uchar*)tra.data)[i*width + j] == 255) points.push_back(Point(j, i));

	vector<Vec4i> lines;
	
	
	HoughLinesP(tra, lines, 1, CV_PI / 180, 10,100,30);

	vector<Vec4f> par;

	for (size_t i = 0; i < lines.size(); i++)
	{
		float slope = (lines[i][1] - lines[i][3])*1.0 / (lines[i][0] - lines[i][2]);
		if (abs(slope) < 0.1) {
			float b = lines[i][1] - slope * lines[i][0];
			par.push_back(Vec4f(slope,b));
			line(tra, Point(lines[i][0], lines[i][1]),
				Point(lines[i][2], lines[i][3]), Scalar(255), 3, 8);
		}
	}
	
	float threshold = 20;
	for (int i = 0; i < points.size();i++) {
		int x = points[i].x;
		int y = points[i].y;
		for (int j = 0; j < par.size(); j++) {
			//限制只提取水平方向直线
			float dis = abs(par[j][0] * x - y + par[j][1]) / sqrt(par[j][0]*par[j][0]+ 1);
			if (dis<threshold) {
				//cout << dis << endl;

				//((uchar*)tra.data)[y*width + x] = 0;
				curve1.push_back(Point(x, y));
				
			}
		}
	}

	cv::imshow("123", tra);
	filename = "E:\\track\\track\\trajectory\\_hough_" + to_string(num) + ".jpg";
	imwrite(filename, tra);
	cv::waitKey();

}
//聚类函数
void eucluster(int num, vector<Point> points, vector<vector<Point>>& clusters, int thre,int minnum,bool save) {
	Mat tra;
	string filename = "E:\\track\\track\\trajectory\\_save_" + to_string(num) + ".jpg";
	tra = imread(filename, 0);
	points.clear();

	int width = tra.cols;
	int height = tra.rows;

	if (num == 1885) {
		for (int i = 0; i < curve1.size(); i++) {
			int x = curve1[i].x;
			int y = curve1[i].y;
			((uchar*)tra.data)[y*width + x] = 0;
		}
	}
	
	for (int i = 0; i < height; i++) 
		for (int j = 0; j < width; j++) 
			if (((uchar*)tra.data)[i*width + j]==255) points.push_back(Point(j, i));
	
	cluster(points, clusters, thre,minnum);
    
	
	//visualize
	Mat show(height, width, CV_8UC3, cv::Scalar::all(0));

	for(int m=0;m<clusters.size();m++)
	for (int i = 0; i < clusters[m].size(); i++) {
		int x = clusters[m][i].x;
		int y = clusters[m][i].y;
		//cout << x << " " << y << endl;
		((uchar*)show.data)[y * width*3 + 3*x] = color[m%6][0];
		((uchar*)show.data)[y * width * 3 + 3 * x+1] = color[m % 6][1];
		((uchar*)show.data)[y * width * 3 + 3 * x+2] = color[m % 6][2];
	}

	if (num == 1885) {
		for (int i = 0; i <curve1.size(); i++) {
			int x = curve1[i].x;
			int y = curve1[i].y;
			//cout << x << " " << y << endl;
			((uchar*)show.data)[y * width * 3 + 3 * x] = color[5][0];
			((uchar*)show.data)[y * width * 3 + 3 * x + 1] = color[5][1];
			((uchar*)show.data)[y * width * 3 + 3 * x + 2] = color[5][2];
		}
	}

	imshow("sad", show);
	waitKey();

	filename = "E:\\track\\track\\trajectory\\_separate_" + to_string(num) + ".png";
	if (save) imwrite(filename, show);
}
//聚类函数中调用的核心函数
void cluster(vector<Point> points, vector<vector<Point>>& clusters, float thre,int minnum) {
	int sizep = points.size();
	int c = 0;

	for (int p = 0; p < sizep; p++) {
		if (points[p].x == -1) continue;
		vector<Point> tem;
		tem.push_back(Point(points[p].x, points[p].y)); 
		points[p].x = points[p].y = -1;
		for (int i = 0; i < tem.size(); i++) {
			int x = tem[i].x;
			int y = tem[i].y;
			for (int j = p + 1; j < sizep; j++) {
				if (points[j].x == -1) continue;
				int xn = points[j].x;
				int yn = points[j].y;
				if (sqrt((xn-x)*(xn-x)*1.0 + (yn-y) * (yn-y)*1.0) < thre) {
					tem.push_back(Point(xn, yn));
					points[j].x = points[j].y = -1;
				}
			}
		}
		if (tem.size() > minnum) {
			clusters.push_back(tem); c++;
		}
	}
	//test
	/*
	cout << c<<"sad";
	for (int i = 0; i < clusters[0].size(); i++)
		cout << clusters[0][i].x << " " << clusters[0][i].y << endl;
	getchar();
    */	
}
//多项式拟合的核心函数
Mat polyfit(std::vector<Point> &chain, int n)
{
	Mat x(chain.size(), 1, CV_32F, Scalar::all(0));
	/* ********【预声明phy超定矩阵】************************/
	/* 多项式拟合的函数为多项幂函数
	* f(x)=a0+a1*x+a2*x^2+a3*x^3+......+an*x^n
	*a0、a1、a2......an是幂系数，也是拟合所求的未知量。设有m个抽样点，则：
	* 超定矩阵phy=1 x1 x1^2 ... ...  x1^n
	*           1 x2 x2^2 ... ...  x2^n
	*           1 x3 x3^2 ... ...  x3^n
	*              ... ... ... ...
	*              ... ... ... ...
	*           1 xm xm^2 ... ...  xm^n
	*
	* *************************************************/
	cv::Mat phy(chain.size(), n, CV_32F, Scalar::all(0));

	for (int i = 0; i<phy.rows; i++)
	{
		float* pr = phy.ptr<float>(i);
		for (int j = 0; j<phy.cols; j++)
		{
			pr[j] = pow(chain[i].y, j);
		}
		x.at<float>(i) = chain[i].x;
	}
	Mat phy_t = phy.t();
	Mat phyMULphy_t = phy.t()*phy;
	Mat phyMphyInv = phyMULphy_t.inv();
	Mat a = phyMphyInv * phy_t;
	a = a * x;
	cout << a << endl;
	return a;
}
//多项式拟合
void myfit(vector<vector<Point>>& clusters, int n) {
	int no;
	Mat coe(clusters.size(), n, CV_32FC1);
	for (int i = 0; i < clusters.size(); i++) {
		Mat tem = polyfit(clusters[i], n);
		for (int j = 0; j < n; j++) ((float*)(coe.data))[i*n + j] = ((float*)(tem.data))[j];
	}

	//visualize
	Mat show(576,720, CV_8UC3, cv::Scalar::all(0));
	for (int i = 0; i < clusters.size(); i++) {
		int ymin = 719; int ymax = 0;
		for (int j = 0; j < clusters[i].size(); j++) {
			if (clusters[i][j].y > ymax) ymax = clusters[i][j].y;
			if (clusters[i][j].y < ymin) ymin = clusters[i][j].y;
		}
		float step = (ymax - ymin)*1.0 / 1000; no = step;
		for (float k = ymin; k < ymax; k += step) {
			double x = 0;
			int y = k;
			for (int l = 0; l < n; l++) { x += pow(y, l)*1.0*((float*)(coe.data))[i*n + l]; } cout << x << endl;;
			if (x< 720&&x>0) {
				((uchar*)(show.data))[y * 720 * 3 + (int)x * 3] = color[i % 6][0];
				((uchar*)(show.data))[y * 720 * 3 + (int)x * 3 + 1] = color[i % 6][1];
				((uchar*)(show.data))[y * 720 * 3 + (int)x * 3 + 2] = color[i % 6][2];
			}
		}
	}

	imshow("fit", show);
	waitKey();
	string filename= "E:\\track\\track\\trajectory\\_fit_" + to_string(no) + ".png";
	imwrite(filename, show);
}