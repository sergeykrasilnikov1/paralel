CXX=pgc++
CXXFLAGS=-fast -ta=tesla:managed -acc -lboost_program_options

NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/

all: exe

exe: laplace2d.o jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $^ ${NVTXLIB}

run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .cpp .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${NVTXLIB}