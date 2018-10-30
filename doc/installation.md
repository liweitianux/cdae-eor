Installation
============

Install the dependencies required to carry out the data simulation
and EoR signal separation.

1. Dependencies directly installed from the repository
   (Debian Linux; testing; amd64):

    ```sh
    $ sudo apt install \
          bison \
          cmake \
          flex \
          g++ \
          gfortran \
          libboost-all-dev \
          libcfitsio-dev \
          libfftw3-dev \
          libgsl-dev \
          libhdf5-dev \
          libopenblas-dev \
          python-dev \
          python-numpy \
          qtbase5-dev \
          wcslib-dev
    ```

2. Install CUDA:

    ```sh
    $ sudo apt install nvidia-cuda-toolkit
    ```

3. Install [CASA Core](https://github.com/casacore/casacore)

    ```sh
    $ git clone https://github.com/casacore/casacore
    $ cd casacore
    # Checkout the v2.4.1, which has been tested with OSKAR.
    # (the latest master as of 2018-10-30 causes error in building OSKAR)
    $ git checkout v2.4.1
    $ mkdir build && cd build
    $ cmake .. \
          -DCMAKE_INSTALL_PREFIX=$HOME/local/casacore \
          -DUSE_OPENMP=ON \
          -DUSE_THREADS=ON \
          -DUSE_FFTW3=ON \
          -DUSE_HDF5=ON \
          -DBUILD_PYTHON=ON
    $ make
    $ make install
    ```

4. Install [OSKAR](http://oskar.oerc.ox.ac.uk/)

    ```sh
    $ git clone https://github.com/OxfordSKA/OSKAR oskar
    $ cd oskar
    $ git checkout 2.7.0
    $ mkdir build && cd build
    $ cmake .. \
          -DCMAKE_INSTALL_PREFIX=$HOME/local/oskar \
          -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
          -DCASACORE_LIB_DIR=$HOME/local/casacore/lib \
          -DCASACORE_INC_DIR=$HOME/local/casacore/include \
          -DFIND_CUDA=ON
    $ make
    $ make install
    $ cd ~/bin && ln -s ../local/oskar/bin/oskar_sim_interferometer .
    ```

5. Install [WSClean](https://sourceforge.net/p/wsclean)

    ```sh
    $ git clone https://git.code.sf.net/p/wsclean/code wsclean-code
    $ cd wsclean-code/wsclean
    $ mkdir build && cd build
    $ cmake .. \
          -DCMAKE_INSTALL_PREFIX=$HOME/local/wsclean \
          -DCMAKE_PREFIX_PATH=$HOME/local/casacore
    $ make
    $ make install
    $ cd ~/bin && ln -s ../local/wsclean/bin/wsclean .
    ```

6. Clone this repo and create the Python virtualenv:

    ```sh
    $ git clone https://github.com/liweitianux/cdae-eor
    $ cd cdae-eor
    $ make venv
    $ source venv/bin/activate
    (venv)$ # ready to go :-)
    ```
