Neural Network Overview
======================

A Neural Network trainer and tester. There are three layers: input, hidden, and output layer. Between layers are neurons, the zeroth neurons are biases with a fixed activation of -1, and the rest are weights with a sigmoid activation function. The neuron's activation takes in as argument the sum of all weights' multiplied.

Explanation of Inputs and Outputs
=================================
- Formats for input/output neural networks:
	- First line has 3 integers separated by spaces, specifying number of input neurons (ni), number of hidden neurons (nh), and number of output neurons (no).
	- The next nh lines (representing nh hidden nodes) contain ni+1 values each, each of them double-precision floating point weights originating from the input layer and ending in the hidden layer. For each line, the first number is the zeroth weight, the bias weight for an input node. The rest are weights going from all inputs to each hidden node. Therefore, there are ni weights plus one bias, for ni+1 weights per hidden node.
	- The next no lines (representing no output nodes) contain nh+1 values each, each of them double-precision floating point weights originating from the hidden layer and ending in theo output layer. These no lines contain the same format as the previous nh lines, but instead of going from input to hidden layer, the nodes go from hidden to output layer.
-Formats for training sets/testing sets:
	- The first line contains three integers separated by spaces, specifying number of training/testing example values (numExamples), number of inputs (ni), and number of outputs (no). ni and no should be identical to values from the neural network.
	- Every line following should have ni double precision floating point inputs, followed by no boolean outputs (either 0 or 1)
-Output results after training
	The output file for testing contains several metrics measuring the network's performance.
		- Overall Accuracy (fraction of examples correctly predicted with respect to the current output category)
		- Precision (of examples predicted to belong to the current category, the fraction that actually below to it)
		- Recall (of examples that actually belong to a category, the fraction predicted to belong)
		- F1 (a technique of combining precision and recall into one metric, always having a value between precision and recall, closer to the lower of the two)
		The output text file has the following format
		- The first no lines contain accuracy, precision, recall, and F1, respectively, for each output category.
		- The last two lines contain accuracy, precision, recall, and F1, respectively, microaveraged (2nd last line) and macroaveraged (last line).
		
		- Note: All double precision values rounded to three decimal places.

User Interface/Instructions
===========================
When running the training program, the user will first be asked to provide an input neural network. Verification of this program involved an input neural network generated through use of a random number generator, but any file of a correct format works. The user will also be prompted for a training file to train the input neural network on, and an output file name to output the trained neural network to. The user will also be prompted for the number of epochs (number of iterations through the training set) and learning rate (how large weights adustments are).

When running the testing program, the user will be asked for an input file (preferably the output of the training file), a testing file, and an output file for outputting metrics.

Primary use
===========
In order to use the files contained in this repository, and test their usability, the following actions are recommended:
- download the .zip file and open in an environment of your choice.
- run the following commands in a shell of your choice: "make train.exe", "make test.exe", "./train.exe", and "./test.exe"
- for the training program, the arguments will be as follows:
	- PIDuntrainedNN.txt
	- pima-indians-diabetes-train.txt
	- 1000
	- .15
	- PIDtrainedNN.txt
- For the testing program, the arguments will be as follows:
	- PIDtrainedNN.txt
	- pima-indians-diabetes-test.txt
	- pima-indians-diabetes-results.txt
-Hopefully, after such commands, the results will be the same as the original ones. Feel free to mess around with/reverse engineer whatever you want. 


Other Recommended Uses
=======================

If one wishes to use this neural network to analyze performance of the neural network, it is recommended to find a relatively large dataset, and split it into two parts. The larger training file should contain the majority of the data, whereas the smaller testing file should have a smaller portion. Inputs should be normalized to their largest value (divide by the largest number in the dataset) so that they are of values between zero and one, and the file should be adjusted to match the aforementioned format. A initial neural network may be created by randomly generating as many numbers as there are weights, and adjusting to match the aforementioned format.

If one wishes to use this neural network to identify outputs of unknown data, it is recommended to have at least a few thousand input weights (number of examples times number of inputs), with corresponding outputs. This will become the training set for a new neural network. Afterwards, it is recommended to have the same inputs for remaining examples to which outputs are unknown. However, it is noteworthy that this is a simple neural network, completed mostly for instructional/learning purposes. Therefore, if one were to seriously want to pursue use of a neural network/artificial intelligence to identify signs of diabetes (as you can do with my test/training set) or a similarly important task, I suggest looking into machine learning repositories or packages more complete and difficult than this.

Thanks for reading this! This is my first project/readme on Github! Please let me know if there are any concerns, or if you have any feedback. Have a fantastic day!
