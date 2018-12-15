#BOOST_INC=/usr/include/boost
BOOST_LIB=`./boost_python_version.sh`


#CXX=g++

target:cwt1d.so

cwt1d.so:pycwt1d.o fftw_blitz.o
	$(CXX) -o $@ $< --shared `python-config --libs` -lgsl -lgslcblas $(BOOST_LIB) -fPIC fftw_blitz.o -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads

pycwt1d.o:pycwt1d.cpp
	$(CXX) -c -o $@ $< `python-config --includes` -fPIC -I ../cwt1d/ -I ../fftw_blitz 

fftw_blitz.o:
	g++ -c ../fftw_blitz/fftw_blitz.cpp -I ../fftw_blitz -o $@ -fPIC


clean:
	rm -f cwt1d.so `find -iname '*.o'`


