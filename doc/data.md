Data Simulation
===============

This doc describes the simulation of SKA *observed* images, which are
used to demonstrate the proposed EoR signal separation method based on
the CDAE and evaluate its performance.

TL;DR
-----
The simulated image cubes used in the paper are provided for download:

* [EoR signal](../data/eor.uvcut_b158c80_n360-cube.fits)
  (md5: bbbd03884960b967c56f05cfc6eba5ff)
* [Galactic emission](../data/gal.uvcut_b158c80_n360-cube.fits)
  (md5: 58663064963d0ee480a1a2d96094e351)
* [Extragalactic point sources](../data/ptr.uvcut_b158c80_n360-cube.fits)
  (md5: fae3216eb477e873f37935c26de10bcc)
* [Radio halos](../data/halos.uvcut_b158c80_n360-cube.fits)
  (md5: 1362542f19b408fdf6a62ba0b7ba2a8a)

All the image cubes cover a sky patch of size 2x2 deg^2 with a pixel
size of 20 arcsec and spans a frequency band of 154-162 MHz with channel
width of 80 kHz.
The size of image cubes is thus 360x360x101.


Sky Maps of Foreground Components
---------------------------------
We use the [FG21sim](https://github.com/liweitianux/fg21sim) software
that we developed based on our previous works
([Wang et al. 2010](http://adsabs.harvard.edu/abs/2010ApJ...723..620W),
[2013](http://adsabs.harvard.edu/abs/2013ApJ...763...90W))
but with significant improvements especially to the simulation of
radio halos.

Follow the instructions there to install the `FG21sim` software and
obtain the necessary template maps.

### Galactic synchrotron emission

1. Extract the template patch from the all-sky HEALPix template by using
   [`get-healpix-patch`](https://github.com/liweitianux/fg21sim/blob/master/bin/get-healpix-patch):

    ```sh
    $ get-healpix-patch --smooth --center 0,-27 --size 1800,1800 \
          haslam408_dsds_Remazeilles2014_ns2048.fits \
          haslam408_eor0_fov10.fits
    $ get-healpix-patch --smooth --center 0,-27 --size 1800,1800 \
          GsyncSpectralIndex_Giardino2002_ns2048.fits \
          specindex_eor0_fov10.fits
    ```

2. Create configuration file (`gsyn.conf`):

    ```ini
    [foregrounds]
    galactic/synchrotron = True

    [sky]
    type = "patch"

        [[patch]]
        # ra, dec [deg]
        xcenter = 0.0
        ycenter = -27.0
        # patch image size
        xsize = 1800
        ysize = 1800
        # pixel size [arcsec]
        pixelsize = 20

    [frequency]
    type = "calc"
    step = 0.08
    start = 154.0
    stop = 162.0

    [galactic]
        [[synchrotron]]
        template = "haslam408_eor0_fov10.fits"
        template_freq = 408.0
        indexmap = "specindex_eor0_fov10.fits"
        add_smallscales = False
        prefix = "gsyn"
        output_dir = "output"
    ```

3. Do the simulation:

    ```sh
    $ fg21sim gsyn.conf
    ```

The simulated sky maps are named `gsyn_<freq>.fits`.


### Galactic free-free emission

1. Extract the template patch from the all-sky HEALPix template:

    ```sh
    $ get-healpix-patch --smooth --center 0,-27 --size 1800,1800 \
          Halpha_fwhm06_ns1024.fits \
          halpha_eor0_fov10.fits
    $ get-healpix-patch --smooth --center 0,-27 --size 1800,1800 \
          SFD_i100_ns1024.fits \
          SFD_eor0_fov10.fits
    ```

2. Create configuration file (`gff.conf`):

    ```ini
    [foregrounds]
    galactic/freefree = True

    [sky]
    type = "patch"

        [[patch]]
        # ra, dec [deg]
        xcenter = 0.0
        ycenter = -27.0
        # patch image size
        xsize = 1800
        ysize = 1800
        # pixel size [arcsec]
        pixelsize = 20

    [frequency]
    type = "calc"
    step = 0.08
    start = 154.0
    stop = 162.0

    [galactic]
        [[freefree]]
        halphamap = "halpha_eor0_fov10.fits"
        dustmap = "SFD_eor0_fov10.fits"
        prefix = "gff"
        output_dir = "output"
    ```

3. Do the simulation:

    ```sh
    $ fg21sim gff.conf
    ```

The simulated sky maps are named `gff_<freq>.fits`.


### Radio halos

1. Create configuration file (`halos.conf`):

    ```ini
    [foregrounds]
    extragalactic/clusters = True

    [sky]
    type = "patch"

        [[patch]]
        # ra, dec [deg]
        xcenter = 0.0
        ycenter = -27.0
        # patch image size
        xsize = 1800
        ysize = 1800
        # pixel size [arcsec]
        pixelsize = 20

    [frequency]
    type = "calc"
    step = 0.08
    start = 154.0
    stop = 162.0

    [extragalactic]
        [[psformalism]]
        dndlnm_outfile = "dndlnm.npz"

        [[clusters]]
        catalog_outfile = "catalog.csv"
        halos_catalog_outfile = "halos.csv"
        halo_dropout = 0
        mass_min = 2e14
        prefix = "halos"
        output_dir = "output"
    ```

2. Do the simulation:

    ```sh
    $ fg21sim halos.conf
    ```

The simulated sky maps are named `halos_<freq>.fits`.


### Extragalactic point sources

This foreground component is simulated by using the
[radio-fg-simu-tools/pointsource](https://github.com/liweitianux/radio-fg-simu-tools/tree/master/pointsource)
tool.  Refer to there for more details to build the tool.

1. Place all the Wilman2008 simulation data at a directory (`wilman2008_db`)
   and generate the list:

    ```sh
    $ ls wilman2008_db/*.txt > wilman2008_db.list
    ```

2. Generate the list of frequencies using the [`freq2z.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/21cm/freq2z.py) tool:

    ```sh
    $ freq2z.py 154:0.08:162 | awk '/^1/ { print $1 }' > freqs.list
    ```

3. Perform the simulation of point sources:

    ```sh
    $ make_ptr_map -o ptr_ -O ptr_sources.csv \
          -i wilman2008_db.list \
          $(cat freqs.list)
    ```

The simulated sky maps of point sources are named `ptr_<freq>.fits`.


Sky Maps of the EoR Signal
--------------------------
The *faint galaxies* simulation case released by the
[Evolution Of 21 cm Structure](http://homepage.sns.it/mesinger/EOS.html)
project.

1. Get the *light travel* cube in redshift range of 5-9.568
   (`delta_T_v3_no_halos__zstart005.00000_zend009.56801_FLIPBOXES0_1024_1600Mpc_lighttravel`).

2. Convert the cube from 21cmFAST binary format to FITS format by using
   [`21cmfast_lightcone.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/21cm/21cmfast_lightcone.py):

    ```sh
    $ 21cmfast_lightcone.py --z-min 5 --z-max 9.56801 \
            delta_T_v3_no_halos__zstart005.00000_zend009.56801_FLIPBOXES0_1024_1600Mpc_lighttravel \
            Tb_lightcone_N1024_L1600_z05.000-09.568.fits
    ```

3. Extract slices at requested frequencies (i.e., redshifts) by using
   [`get_slice_zfreq.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/21cm/get_slice_zfreq.py):

    ```sh
    $ mkdir deltaTb_slices && cd deltaTb_slices
    $ get_slice_zfreq.py \
          -i Tb_lightcone_N1024_L1600_z05.000-09.568.fits
          -f $(cat freqs.list)
    ```

   The extracted slices are named as `deltaTb_f<freq>_z<redshift>.fits`.

4. Tile the slices to match the FoV and pixel size by using
   [`tile_slice.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/21cm/tile_slice.py):

    ```sh
    $ mkdir deltaTb_tiled && cd deltaTb_tiled
    $ for slice in ../deltaTb_slices/*.fits; do \
          tile_slice.py -f 158 -F 10.0 -N 1800 -i ${slice}; \
      done
    ```


Observational Simulation
------------------------
