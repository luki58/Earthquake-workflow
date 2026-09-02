
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CC = g++-15
	CC_OPT = -fopenmp -O3 -pedantic -W -Wall -Wextra -std=c++0x
else
	CC = g++
	CC_OPT = -fopenmp -O3 -pedantic -W -Wall -Wextra -std=c++0x
endif


CONFIG_LIBTIFF = `pkg-config --cflags --libs libtiff-4`
# In case pkg-config doesn't work:
# CONFIG_LIBTIFF = -I/usr/local/Cellar/libtiff/4.1.0/include -L/usr/local/Cellar/libtiff/4.1.0/lib -ltiff

all: FindGrains

FindGrains: FindGrains.cpp Makefile
	$(CC) $(CC_OPT) -o FindGrains FindGrains.cpp $(CONFIG_LIBTIFF)

clean:
	rm -f *.o FindGrains


