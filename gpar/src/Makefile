# Makefile for coordtrans

CC = gcc
CFLAGS = -g3 -ggdb -O0 -fno-inline -Wall -Wextra 
#CFLAGS = $(CFLAGS) -pedantic
#CFLAGS = $(CFLAGS) -lccmalloc –ldl
#CFLAGS = $(CFLAGS) -lefence

DEPS = \
	array.o

# FFT = $(shell python -c "import numpy, os; print os.path.dirname(numpy.fft.__file__)")


signal.so: $(DEPS)
	$(CC) -shared $(CFLAGS) -lm -o array.so $(DEPS)

install: signal.so
	cp array.so ../lib/libarray.so

clean:
	find ./ -name "*.o" -exec rm {} \;
	rm -f *.so
