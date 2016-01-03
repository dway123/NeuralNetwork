#include <iostream>
#include <string>
#include <cstdlib>
using namespace std;
#include "nn.h"

int main(){
	cout << "Welcome to the Neural Network testing program." << endl;
	
	nn neuralnetwork;
	if(!neuralnetwork.read()){
		return 0;
	}
	if(!neuralnetwork.test()){
		return 0;
	}
	cout << "Success" << endl;
}
