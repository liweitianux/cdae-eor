all: lib/libblitz.so

lib/libblitz.so: blitz-0.9
	-mkdir lib
	cd $< && \
	    ./configure --prefix=/tmp/$<-install \
	        --libdir=$(PWD)/lib --include=$(PWD)/include \
	        --disable-fortran --disable-doxygen \
	        --disable-dot --disable-html-docs \
	        --disable-static --enable-shared; \
	    make && \
	    make install && \
	    rm -rf /tmp/$<-install

blitz-0.9: blitz-0.9.tar.gz
	tar xf $<

clean:
	rm -rf blitz-0.9

cleanall: clean
	rm -rf include/blitz include/random
	rm -rf lib/pkgconfig lib/libblitz.*
