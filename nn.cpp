#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <math.h> //for exp(x)/e^x
#include <iomanip> //std::setprecision

using namespace std;

#include "nn.h"

//SIG: returns sigmoid(x)
double nn::sig(double x){
	return 1/(1+exp(-x));
}

//SIGP: returns sigmoid'(x)
double nn::sigp(double x){
	return sig(x)*(1-sig(x));
}

double nn::accuracy(int a, int b, int c, int d){
	return (((double)a+d)/(a+b+c+d));
}

double nn::precision(int a, int b){
	return ((double)a/(a+b));
}

double nn::recall(int a, int c){
	return ((double)a/(a+c));
}

double nn::F1(int a, int b, int c, int d){
	return ((2*(double)precision(a,b)*recall(a,c))/(precision(a,b)+recall(a,c)));
}

//NN: empty constructor
nn::nn(){
}

//READ: import a file containing data representing a neural network
bool nn::read(){
	bool testing = false;
	
	string inputFileName;
	cout << "Enter name of input file: " << endl;
	cin >> inputFileName;
	
	ifstream file(inputFileName.c_str());
	if(!file){
		cout << "Cannot open the file" << endl;
		return false;
	}
	
	file >> ni; //number of input nodes
	file >> nh; //number of hidden nodes
	file >> no; //number of output nodes
	
	if(testing){
		cout << "Ni = " << ni << "\nNh = " << nh << "\nNo = " << no << endl;
		cout << "Weights for input to hidden nodes" << endl;
	}
	
	//resize the hidden node vector w
	w.resize(nh);
	for (int j = 0; j < nh; j++){
		w[j].resize(ni+1);
	}
	
	//Nh lines specify weights of edges from input to hidden nodes
	//each of these lines has  Ni+1 weights
	for(int i = 0; i < nh; i++){
		if(testing){cout << "\nHidden node #" << i << ": ";}
		for(int j = 0; j < ni+1; j++){ //on each line: bias weight, then hidden node weights
			file >> w[i][j]; //set the weights
			if(testing){cout << "\t" << w[i][j];}
		}
	}
	
	//resize the vector bw
	bw.resize(no);
	for (int j = 0; j < no; j++){
		bw[j].resize(nh+1);
	}
	
	//next No lines specify weights from hidden to output nodes
	for(int i = 0; i < no; i++){
		if(testing){cout << "\nOutput node #" << i << ": ";}
		for(int j = 0; j < nh+1; j++){ //on each line: bias weight, then output node weights
			file >> bw[i][j]; //set the weights
			if(testing){cout << "\t" << bw[i][j];}
		}
	}
	if(testing){cout << endl;}
	return true;
}

//WRITE: outputs a file containing the data of the neural network, in the same format
bool nn::write(){
	string outputFileName;
	cout << "Enter name of output file: " << endl;
	cin >> outputFileName; 
	
	ofstream output(outputFileName.c_str());
	
    if(!output){
        cout << "Cannot open output file" << endl;
        return false;
    }
	
	output << ni << " " << nh << " " << no << endl; //first line
	
	//Nh lines specify weights of edges from input to hidden nodes
	//each of these lines has  Ni+1 weights
	output << setprecision(3) << fixed;
	for(int i = 0; i < nh; i++){
		for(int j = 0; j < ni+1; j++){ //on each line: bias weight, then hidden node weights
			output << w[i][j]; //set the weights
			if(j!=ni){output << " ";}
		}
		output << endl;
	}
	
	//next No lines specify weights from hidden to output nodes
	output << setprecision(3) << fixed;
	for(int i = 0; i < no; i++){
		for(int j = 0; j < nh+1; j++){ //on each line: bias weight, then output node weights
			output << bw[i][j]; //set the weights
			if(j!=nh){output << " ";}
		}
		output << endl;
	}
	return true;
}

//TRAIN: trains the neural network using a training set file
bool nn::train(){
	bool testing = false;
	
	//open the file
	string trainFileName;
	cout << "Enter name of training set file: " << endl;
	cin >> trainFileName;
	
	ifstream file(trainFileName.c_str());
	if(!file){
		cout << "Cannot open the file" << endl;
		return false;
	}
	
	//get the number of epochs and learning rate for the neural network
	int epochs = 0;
	double learnRate = 0;
	cout << "Enter the number of epochs: " << endl;
	cin >> epochs;
	cout << "Enter the learning rate: " << endl;
	cin >> learnRate;
	if(!epochs){
		cout << "An invalid value, " << epochs << ", was entered for the number of epochs." << endl;
		return false;
	}
	if(!learnRate){
		cout << "An invalid value, " << learnRate << ", was entered for the learning rate. " << endl;
	}
	
	int temp;
	int numExamples;
	
	file >> numExamples; //number of examples
	file >> temp; //number of input nodes
	if(temp != ni){
		cout << "The provided number of input nodes, " << temp << ", was different from the original, " << ni << endl;
	}
	file >> temp; //number of output nodes
	if(temp != no){
		cout << "The provided number of output nodes, " << temp << ", was different from the original, " << no << endl;
	}
	if(testing){cout << "Epochs = " << epochs << "\nLearning rate = " << learnRate << "\nNumber of examples = " << numExamples << endl;}
	
	exIn.resize(numExamples); //exIn is numExamples by ni
	for (int j = 0; j < numExamples; j++){
		exIn[j].resize(ni);
	}
	
	exOut.resize(numExamples); //exOut is numExamples by no
	for (int j = 0; j < numExamples; j++){
		exOut[j].resize(no);
	}
	
	//put training examples into the exIn and exOut arrays
	for(int i = 0; i < numExamples; i++){ //parse numExamples examples, each with ni inputs and then no outputs
		for(int j = 0; j < ni; j++){
			file >> exIn[i][j];
		}
		for(int j = 0; j < no; j++){
			file >> exOut[i][j];
		}
	}
	//BACK PROPOGATION LEARNING
	//referenced Figure 18.24 from textbook, as annotated by Professor Sable
	//The actual training is done in these following loops, using examples in the exIn and exOut arrays we just made
	vector<double> deltah(nh,0); //a local vector of errors
	vector<double> deltab(no,0); //a local vector of errors for output nodes
	vector<double> inh(nh,0); //hidden inputs
	vector<double> inb(no,0); //output inputs
	vector<double> a1(ni,0);
	vector<double> a2(nh,0);
	vector<double> b(no,0);
	double sum = 0;
	//weights already initialized in the nn
	for(int ep = 0; ep < epochs; ep++){ //the repeat...until loop in the pseudo code
		for(int ex = 0; ex < numExamples; ex++){ //for each example(x,y) in examples do
			//pretty much everything is within the 2 above loops
			//REINITIALIZE reused variables
			for(int i = 0; i < nh; i++){
				inh.at(i) = 0; //reinitialize vector inh
			}
			for(int i = 0; i < no; i++){
				inb.at(i) = 0; //reinitialize vector inb
			}
			for(int i = 0; i < ni; i++){ //copying input vector of single training example to input nodes of network
					a1.at(i) = exIn[ex][i];
			}
			//IMPLEMENTATION below
			//PROPOGATE FORWARD
			//for the original weights vector hidden layer
			//for each hidden node j, adds every input i's contribution with w[i][j] so that inh[j] has the hidden node's input value.
			//then a2[j], the hidden node j, is augmented to have the output of the hidden layer or input to the output layer
			for(int j = 0; j < nh ; j++){ //hidden nodes
				for(int i = 0; i < ni+1; i++){ //inputs
					if(i == 0){
						inh.at(j) += w[j][i] * -1;
					}
					else{
						inh.at(j) += w[j][i]*a1.at(i-1); //weight*input
					}
				}
				a2.at(j) = sig(inh.at(j)); //hidden layer output/output layer input
			}
			//for the weights of the output vector
			//for each output node j, adds hidden node's contribution in[j] with bw[i][j] so that in[j[ has the output node's value
			//then a[j], the hidden node j, is augmented???
			for(int j = 0; j < no; j++){
					for(int i = 0; i < nh + 1; i++){
						if(i == 0){
							inb.at(j) += bw[j][i] * -1; //offset
						}
						else{
							inb.at(j) += bw[j][i]*a2.at(i-1);
						}
					}
					b.at(j) = sig(inb.at(j));
			}
			//BACK PROPOGATION of deltas
			//for each output node j in the output layer, find the error delta of the node, to be 
			//equal to the sigp of the expected output in[j] * (the actual output minus the activation)
			for(int j = 0; j < no; j++){
				deltab.at(j) = sigp(inb.at(j))*(exOut[ex][j] - b.at(j)); //computing the delta of the jth output node.
			}
			//for the weights hidden layer vector
			//for each input node i, the error at the input node delta[i] = sigp(in[i])*the sum of all errors at j, the hidden layer
			//therefore the error there equals the output node's value times the error delta.
			for(int i = 1; i < nh+1; i++){ //bypass the bias weight
				sum = 0;
				for(int j = 0; j < no; j++){
					sum += bw[j][i]*deltab.at(j);
				}
				deltah[i-1] = sigp(inh.at(i-1))*sum; //from inh[0] to inh[nh]
			}
			
			//AUGMENTING WEIGHTS
			//for each weight w[i][j] do
			//hidden weights
			for(int i = 0; i < ni + 1; i++){
				for(int j = 0; j < nh; j++){
					if(i == 0){
						w[j][i] = w[j][i] + learnRate * -1 * deltah.at(j);
					}
					else{
						w[j][i] = w[j][i] + learnRate * a1.at(i-1) * deltah.at(j);
					}
				}
			}
			//output weights
			for(int i = 0; i < nh+1; i++){
				for(int j = 0; j < no; j++){
					if(i == 0){
						bw[j][i] = bw[j][i] + learnRate * -1 * deltab.at(j);
					}
					else{
						bw[j][i] = bw[j][i] + learnRate * a2.at(i-1) * deltab.at(j);
					}
				}
			}			
		}
	}
	return true;
}

//TEST: tests a (trained) neural network using an input file
bool nn::test(){
	bool testing = false;
	
	//open the file
	string testFileName;
	cout << "Enter name of testing set file: " << endl;
	cin >> testFileName;
	
	ifstream file(testFileName.c_str());
	if(!file){
		cout << "Cannot open the file" << endl;
		return false;
	}
	
	//reading in the file
	int temp;
	int numExamples;
	
	file >> numExamples; //number of examples
	file >> temp; //number of input nodes
	if(temp != ni){
		cout << "The provided number of input nodes, " << temp << ", was different from the original, " << ni << endl;
	}
	file >> temp; //number of output nodes
	if(temp != no){
		cout << "The provided number of output nodes, " << temp << ", was different from the original, " << no << endl;
	}
	
	exIn.resize(numExamples); //exIn is numExamples by ni
	for (int j = 0; j < numExamples; j++){
		exIn[j].resize(ni);
	}
	
	exOut.resize(numExamples); //exOut is numExamples by no
	for (int j = 0; j < numExamples; j++){
		exOut[j].resize(no);
	}
	
	for(int i = 0; i < numExamples; i++){ //parse numExamples examples, each with ni inputs and then no outputs
		for(int j = 0; j < ni; j++){
			file >> exIn[i][j];
		}
		for(int j = 0; j < no; j++){
			file >> exOut[i][j];
		}
	}
	
	//put training examples into the exIn and exOut arrays
	for(int i = 0; i < numExamples; i++){ //parse numExamples examples, each with ni inputs and then no outputs
		for(int j = 0; j < ni; j++){
			file >> exIn[i][j];
		}
		for(int j = 0; j < no; j++){
			file >> exOut[i][j];
		}
	}
	
	//initialize variables for test results
	vector<int> aa(no,0);
	vector<int> bb(no,0);
	vector<int> cc(no,0);
	vector<int> dd(no,0);
	
	//FORWARD PROPOGATION TESTING
	//referenced Figure 18.24 from textbook, as annotated by Professor Sable
	//Essentially the same as back propagation strategy, but with 1 epoch and no back propagation
	vector<double> inh(nh,0); //hidden inputs
	vector<double> inb(no,0); //output inputs
	vector<double> a1(ni,0);
	vector<double> a2(nh,0);
	vector<double> b(no,0);
	double sum = 0;
	//weights already initialized in the nn
	for(int ex = 0; ex < numExamples; ex++){ //for each example(x,y) in examples do
		//pretty much everything is within the 2 above loops
		//REINITIALIZE reused variables
		for(int i = 0; i < nh; i++){
			inh.at(i) = 0; //reinitialize vector inh
		}
		for(int i = 0; i < no; i++){
			inb.at(i) = 0; //reinitialize vector inb
		}
		for(int i = 0; i < ni; i++){ //copying input vector of single training example to input nodes of network
				a1.at(i) = exIn[ex][i];
		}
		//IMPLEMENTATION below
		//PROPOGATE FORWARD
		//for the original weights vector hidden layer
		//for each hidden node j, adds every input i's contribution with w[i][j] so that inh[j] has the hidden node's input value.
		//then a2[j], the hidden node j, is augmented to have the output of the hidden layer or input to the output layer
		for(int j = 0; j < nh ; j++){ //hidden nodes
			for(int i = 0; i < ni+1; i++){ //inputs
				if(i == 0){
					inh.at(j) += w[j][i] * -1;
				}
				else{
					inh.at(j) += w[j][i]*a1.at(i-1); //weight*input
				}
			}
			a2.at(j) = sig(inh.at(j)); //hidden layer output/output layer input
		}
		//for the weights of the output vector
		//for each output node j, adds hidden node's contribution in[j] with bw[i][j] so that in[j[ has the output node's value
		//then a[j], the hidden node j, is augmented???
		for(int j = 0; j < no; j++){
				for(int i = 0; i < nh + 1; i++){
					if(i == 0){
						inb.at(j) += bw[j][i] * -1; //offset
					}
					else{
						inb.at(j) += bw[j][i]*a2.at(i-1);
					}
				}
				b.at(j) = sig(inb.at(j));
		}
		//GET ERRORS AND STATS FOR EACH OUTPUT
		for(int j = 0; j < no; j++){
			b.at(j) = (int)(round(b.at(j))); //round outputs to the nearest value
			//cout << b.at(j) << endl;
			if(b.at(j) && exOut[ex][j]){ //if the predicted and expected both equal 1
				aa.at(j)++;
			}
			else if(b.at(j) && !exOut[ex][j]){ //if predicted = 1 and expected = 0
				bb.at(j)++;
			}
			else if(!b.at(j) && exOut[ex][j]){ //if predicted = 0 and expected = 1
				cc.at(j)++;
			}
			else if(!b.at(j) && !exOut[ex][j]){//if predicted and expected both equal 1
				dd.at(j)++;
			}
			else{
				cout << "ERROR: 404 you shouldn't get here" << endl;
			}
		}	
	}
	
	int aaa = 0;
	int bbb = 0;
	int ccc = 0;
	int ddd = 0;
	
	for(int i = 0; i < no; i++){ //for macro averages
		aaa += aa.at(i);
		bbb += bb.at(i);
		ccc += cc.at(i);
		ddd += dd.at(i);
	}
	
	//output
	string outputFileName;
	cout << "Enter name of tests results file: " << endl;
	cin >> outputFileName;
	
	ofstream output(outputFileName.c_str());
	
    if(!output){
        cout << "Cannot open output file" << endl;
        return false;
    }
	
	//for each output
	output << setprecision(3) << fixed;
	vector<double> accuracyVec(no,0);
	vector<double> precisionVec(no,0);
	vector<double> recallVec(no,0);
	vector<double>F1Vec(no,0);
	for(int i = 0; i < no; i++){
		output << aa.at(i) << " " << bb.at(i) << " " << cc.at(i) << " " << dd.at(i) << " ";
		accuracyVec.at(i) = accuracy(aa.at(i),bb.at(i),cc.at(i),dd.at(i));
		precisionVec.at(i) = precision(aa.at(i),bb.at(i));
		recallVec.at(i) = recall(aa.at(i),cc.at(i));
		F1Vec.at(i) = F1(aa.at(i),bb.at(i),cc.at(i),dd.at(i));
		output << accuracyVec.at(i) << " " << precisionVec.at(i) << " " << recallVec.at(i) << " " << F1Vec.at(i) << endl;
	}
	
	//macroaveraging
	output << accuracy(aaa,bbb,ccc,ddd) << " " << precision(aaa,bbb) << " " << recall(aaa,ccc) << " " << F1(aaa,bbb,ccc,ddd) << endl;
	
	//microaveraging
	double accuracyAvg = 0;
	double precisionAvg = 0;
	double recallAvg = 0;
	double F1Avg = 0;
	
	for(int i = 0; i < no; i++){
		accuracyAvg += accuracyVec.at(i);
		precisionAvg += precisionVec.at(i);
		recallAvg += recallVec.at(i);
		F1Avg += F1Vec.at(i);
	}
	accuracyAvg = (double)accuracyAvg/(double)no;
	precisionAvg = (double)precisionAvg/(double)no;
	recallAvg = (double)recallAvg/(double)no;
	F1Avg = 2*precisionAvg*recallAvg/(precisionAvg+recallAvg);

	output << accuracyAvg << " " << precisionAvg << " " << recallAvg << " " << F1Avg;
	
	return true;
}