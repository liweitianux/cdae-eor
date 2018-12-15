CXX?=		g++
CXXFLAGS?=	-Wall -fPIC

PY?=		python3
PY_VERSION:=	$(shell $(PY) -V | cut -c8-10)
INC_PY:=	$(shell $(PY)-config --includes)
LIB_PY:=	$(shell $(PY)-config --libs)
LIB_BOOST?=	-lboost_python$(subst .,,$(PY_VERSION)) \
		-lboost_numpy$(subst .,,$(PY_VERSION))

INC_GSL?=	$(shell gsl-config --cflags)
LIB_GSL?=	$(shell gsl-config --libs)

INC_FIO?=	-I../deps/fio/include
LIB_FIO?=	-L../deps/fio/lib -lfio -lcfitsio -lblitz
LIB_FFTW?=	-lfftw3_threads -lfftw3f_threads -lfftw3 -lfftw3f

all: cwt1d.so

cwt1d.so: pycwt1d.o fftw_blitz.o
	$(CXX) $(CXXFLAGS) --shared -o $@ $^ \
	    $(LIB_PY) $(LIB_GSL) $(LIB_BOOST) $(LIB_FFTW)

pycwt1d.o: pycwt1d.cpp cwt1d_wavelets.hpp cwt1d.hpp
	$(CXX) $(CXXFLAGS) $(INC_PY) $(INC_GSL) $(INC_FIO) -c $<

fftw_blitz.o: fftw_blitz.cpp fftw_blitz.hpp
	$(CXX) $(CXXFLAGS) $(INC_FIO) -c $<

clean:
	rm -f *.o *.so

test:
	echo $(PY) $(PY_VERSION)
	echo $(INC_PY)
	echo $(LIB_PY)
	echo $(LIB_BOOST)
