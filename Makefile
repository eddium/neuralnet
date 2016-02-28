PROG = neuralnet
CC = g++
CPPFLAGS = -Wall -Werror -std=c++1y
OBJS = main.o matrix.o nn.o

$(PROG): $(OBJS)
	$(CC) $(OBJS) -o $(PROG)

main.o: main.cpp
	$(CC) $(CPPFLAGS) -c main.cpp

matrix.o: matrix.cpp matrix.hpp
	$(CC) $(CPPFLAGS) -c matrix.cpp

nn.o: nn.cpp nn.hpp
	$(CC) $(CPPFLAGS) -c nn.cpp

clean:
	rm -rf *.o
	rm -f $(PROG)