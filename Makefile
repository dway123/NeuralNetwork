test.exe: test.o nn.o
	g++ -o test.exe test.o nn.o  -std=c++11

train.exe: train.o nn.o
	g++ -o train.exe train.o nn.o  -std=c++11
	
train.o: train.cpp
	g++ -c train.cpp  -std=c++11
	
test.o: test.cpp
	g++ -c test.cpp  -std=c++11

nn.o: nn.cpp nn.h
	g++ -c nn.cpp  -std=c++11
	
debug:
	g++ -g  -std=c++11 -o traindebug.exe train.cpp nn.cpp
	
clean:
	rm -f *.exe *.o *.stackdump *~

backup:
	test -d backups || mkdir backups
	cp *.cpp backups
	cp *.h backups
	cp Makefile backups
