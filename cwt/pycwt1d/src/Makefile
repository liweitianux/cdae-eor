CXX?=		g++
CXXFLAGS?=	-Wall -fPIC

INC_PY:=	$(shell python3-config --includes)
LIB_PY:=	$(shell python3-config --libs)
LIB_BOOST?=	-lboost_python3 -lboost_numpy3
INC_GSL?=	$(shell gsl-config --cflags)
LIB_GSL?=	$(shell gsl-config --libs)
LIB_FFTW?=	-lfftw3_threads -lfftw3f_threads -lfftw3 -lfftw3f
INC_BLITZ?=	-I../deps/include

all: cwt1d.so

cwt1d.so: pycwt1d.o fftw_blitz.o
	$(CXX) $(CXXFLAGS) --shared -o $@ $^ \
	    $(LIB_PY) $(LIB_GSL) $(LIB_BOOST) $(LIB_FFTW)

pycwt1d.o: pycwt1d.cpp cwt1d_wavelets.hpp cwt1d.hpp
	$(CXX) $(CXXFLAGS) $(INC_PY) $(INC_GSL) $(INC_BLITZ) -c $<

fftw_blitz.o: fftw_blitz.cpp fftw_blitz.hpp
	$(CXX) $(CXXFLAGS) $(INC_BLITZ) -c $<

clean:
	rm -f *.o cwt1d.so
