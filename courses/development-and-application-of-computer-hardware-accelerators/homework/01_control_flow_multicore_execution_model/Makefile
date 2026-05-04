COMPILER := gcc
COMPILER_FLAGS := -O2 -fopenmp
EXECUTABLE := bin/matrix_multiplication
SOURCE_FILE := matrix_multiplication.c

NUMBER_OF_RUNS := 20

clean:
	rm -f $(EXECUTABLE)

build: 
	$(COMPILER) $(SOURCE_FILE) $(COMPILER_FLAGS) -o $(EXECUTABLE)

matrix_multiplication_run_512_512: clean build
	./$(EXECUTABLE) $(NUMBER_OF_RUNS) 512 512

matrix_multiplication_run_1024_1024: clean build
	./$(EXECUTABLE) $(NUMBER_OF_RUNS) 1024 1024