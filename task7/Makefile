CXX=pgc++
CXXFLAGS=-fast -acc -ta=tesla -Minfo=accel \
         -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
         -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
         -lboost_program_options
LDFLAGS=-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
        -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
        -lcublas -lcudart

all: exe

exe: laplace2d.o jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run: exe
	CUDA_VISIBLE_DEVICES=3 ./exe --n 512 --iter 10000000 --err 1e-6

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDFLAGS)
