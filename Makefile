################################################################# 
# Makefile 
# 需要显式指定各个编译目录的位置； 
################################################################# 

CC=nvcc -O3 
# CC = nvcc -O3
RM = rm -rf
MAKE = make

DIR_INC = ./include
DIR_SRC = ./src
DIR_BIN = ./bin

SRC = $(wildcard ${DIR_SRC}/*.cu)  
OBJ = $(patsubst %.cu,${DIR_BIN}/%.o,$(notdir ${SRC})) 

TARGET = main
BIN_TARGET = ${DIR_BIN}/${TARGET}

CFLAGS = -I${DIR_INC}
CFLAGS += -Xcompiler -fopenmp
CFLAGS += -arch=sm_75
CFLAGS += -lcufft

.PHONY:all clean

all:${BIN_TARGET}

# ${BIN_TARGET}:${SRC} main.c
# 	$(CC) $^ -o $@ -I${DIR_INC} 

${BIN_TARGET}:${OBJ} bin/main.o
	$(CC)  $(CFLAGS)  $^ -o ./app/dmselect

${DIR_BIN}/%.o:$(DIR_SRC)/%.cu
	$(CC) $(CFLAGS) -c $^ -o $@ 


bin/main.o:main.cu
	$(CC) $(CFLAGS) -c main.cu -o bin/main.o
clean:
	find ${DIR_BIN} . -name '*.o' -exec $(RM) '{}' \;
#	find ${DIR_BIN} -name '*.o' | xargs rm -rf
	$(RM) ${BIN_TARGET}
