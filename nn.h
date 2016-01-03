#ifndef _NN_H
#define _NN_H
#include<vector>
using namespace std;

//include libraries, etc.

class nn{
public:
	
	nn(); //constructor does... nothing?
	
	bool read(); //reads the neural network from a file
	bool write(); //writes the neural network to a file
	
	bool testtrainread(); //reads a file in for testing or training
	bool train(); //trains the neural network using a training set file
	bool test(); //tests a trained neural network using a test file

	double sig(double x);//returns sigmoid(x)
	double sigp(double x);//returns sigmoid'(x)
	
	double accuracy(int a, int b, int c, int d);
	double precision(int a, int b);
	double recall(int a, int c);
	double F1(int a, int b, int c, int d);
	
	void display(); //displays the neural network

private:
	int ni = 0;
	int nh = 0;
	int no = 0;
	
	vector<vector<double> > w;
	vector<vector<double> > bw;
	//store the training examples
	vector<vector<double> > exIn; //stores an example's inputs
	vector<vector<double> > exOut; //stores an example's outputs
	
	//double a[ni]; //input vector
	//double w[ni+1][nh]; //hidden node weight vector
	//double bw[nh+1][no]; //weights for output nodes
	//double b[no]; //output vector
	//double b[no]; //output vector
	//double b[no]; //output vector
};
#endif //_NN_H