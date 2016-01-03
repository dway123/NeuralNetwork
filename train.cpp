#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <math.h> //for exp(x)/e^x

using namespace std;

#include "nn.h"

int main(){
	cout << "Welcome to the Neural Network training program." << endl;
	nn neuralnetwork;
	if(!neuralnetwork.read()){
		return 0;
	}
	if(!neuralnetwork.train()){
		return 0;
	}
	if(!neuralnetwork.write()){
		return 0;
	}
	cout << "Success" << endl;
}