Data Simulation
===============

This doc describes the simulation of SKA *observed* images, which are
used to demonstrate the proposed EoR signal separation method based on
the CDAE and evaluate its performance.

TL;DR
-----
The simulated image cubes **with realistic instrumental effects** integrated
are provided for download:

* [EoR signal](../data/eor.uvcut_b158c80_n360-cube.fits)
  (md5: bbbd03884960b967c56f05cfc6eba5ff)
* [Galactic emission](../data/gal.uvcut_b158c80_n360-cube.fits)
  (md5: 58663064963d0ee480a1a2d96094e351)
* [Extragalactic point sources](../data/ptr.uvcut_b158c80_n360-cube.fits)
  (md5: fae3216eb477e873f37935c26de10bcc)
* [Radio halos](../data/halos.uvcut_b158c80_n360-cube.fits)
  (md5: 1362542f19b408fdf6a62ba0b7ba2a8a)

Each files is about 50 MiB in size.
All the image cubes cover a sky patch of size 2x2 deg^2 with a pixel
size of 20 arcsec and spans a frequency band of 154-162 MHz with channel
width of 80 kHz.
The size of image cubes is thus 360x360x101.

Also used in the paper for comparison, the **ideal skymaps** (i.e., for which
the observational simulations are performed) are also provided:

* [EoR signal (skymap)](../data/eor_b158c80_n360-cube.fits)
  (md5: 7446c159d8c0b6964a9eead881db3995)
* [Galactic emission (skymap)](../data/gal_b158c80_n360-cube.fits)
  (md5: 3c67c57895653701d62e94a077b82f46)
* [Extragalactic point sources (skymap)](../data/ptr_b158c80_n360-cube.fits)
  (md5: f1e1e4e827ff5b6155bf5c9dbb1dd75b)
* [Radio halos (skymap/bzipped)](../data/halos_b158c80_n360-cube.fits.bz2)
  (md5: 7ce76b8f2c26393951f6ed05c300509f)


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

### A. Galactic synchrotron emission

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


### B. Galactic free-free emission

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


### C. Radio halos

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
The maps of the EoR signal are created by using the data of the
*faint galaxies* simulation case released by the
[Evolution Of 21 cm Structure](http://homepage.sns.it/mesinger/EOS.html)
project.

1. Get the *light travel* cube in redshift range of 5-9.568, i.e., the file named
   `delta_T_v3_no_halos__zstart005.00000_zend009.56801_FLIPBOXES0_1024_1600Mpc_lighttravel`.

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
By employing the latest
[SKA1-Low layout configuration](https://astronomers.skatelescope.org/wp-content/uploads/2016/09/SKA-TEL-SKO-0000422_02_SKA1_LowConfigurationCoordinates-1.pdf)
(released on 2016 May 21),
the [OSKAR](https://github.com/OxfordSKA/OSKAR) simulator is used to
perform the observational simulation to generate the visibility data
for each sky map.
Then the visibility data is imaged by the
[WSClean](https://sourceforge.net/projects/wsclean/) imager to create
the simulated SKA images.

1. Generate the *telescope model* for the OSKAR simulator by using
   [`make-ska1low-model`](https://github.com/liweitianux/fg21sim/blob/master/bin/make-ska1low-model):

   ```sh
   $ make-ska1low-model -o ska1low.tm
   ```

   Or use the pre-generated one by me at
   [atoolbox/astro/oskar/telescopes/ska1low.tm](https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/telescopes/ska1low.tm).

2. Convert the sky maps into *sky models* for OSKAR by using
   [`fits2skymodel.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/fits2skymodel.py):

   ```sh
   $ for f in <skymaps-dir>/*<freq>.fits; do \
         fits2skymodel.py -o skymodel ${f}; \
         outfile=skymodel/$(basename ${f%.fits}.osm); \
         echo "<freq> ${outfile}" >> skymodel.list; \
     done
   ```

   For the Galactic emission, the sky maps of synchrotron and free-free
   emissions are combined before converting to the OSKAR sky models.

   As for the extragalactic point sources, the bright sources with a
   158 MHz flux density greater than 10 mJy are removed first, which
   corresponds to pixels with value greater than ~1400 K are removed.

   ```sh
   $ fits2skymodel.py --pixel-size 20 --max-value 1400 \
         -f 158 -o skymodel --create-mask mask.fits \
         <skymaps-dir>/ptr_158.00.fits
   $ rm skymodel/ptr_158.00.osm

   $ for f in <skymaps-dir>/*<freq>.fits; do \
         fits2skymodel.py --pixel-size 20 -f <freq> \
             --mask mask.fits -o skymodel ${f}; \
         outfile=skymodel/$(basename ${f%.fits}.osm); \
         echo "<freq> ${outfile}" >> skymodel.list; \
     done
   ```

3. Prepare the base OSKAR configuration file (`oskar.ini`):

   ```ini
   [General]
   app=oskar_sim_interferometer

   [simulator]
   max_sources_per_chunk=524288
   double_precision=false
   use_gpus=true
   keep_log_file=true
   write_status_to_log_file=true

   [sky]
   advanced/apply_horizon_clip=false
   oskar_sky_model/file=

   [telescope]
   aperture_array/array_pattern/enable=true
   aperture_array/element_pattern/dipole_length=0.5
   input_directory=telescopes/ska1low.tm
   aperture_array/element_pattern/dipole_length_units=Wavelengths
   aperture_array/element_pattern/functional_type=Dipole
   pol_mode=Scalar
   normalise_beams_at_phase_centre=true
   allow_station_beam_duplication=true
   station_type=Aperture array

   [observation]
   phase_centre_ra_deg=0.0
   phase_centre_dec_deg=-27.0
   start_time_utc=2000-01-01T06:30:00.000
   length=21600.0
   num_time_steps=72
   num_channels=1
   start_frequency_hz=

   [interferometer]
   uv_filter_max=max
   time_average_sec=300.0
   uv_filter_units=Wavelengths
   channel_bandwidth_hz=80000.0
   uv_filter_min=min
   oskar_vis_filename=
   ms_filename=
   ```

4. Do observational simulations by using
   [`run_oskar.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/run_oskar.py):

   ```sh
   $ run_oskar.py -c oskar.ini -l skymodel.list -o visibility
   ```

5. Create images by using
   [`wsclean.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/wsclean.py),
   which wraps on WSClean to be easier to use:

   ```sh
   $ mkdir images && cd images
   $ wsclean.py --threshold <threshold> --weight natural \
         --uv-range 30:1000 --size 1800 --pixelsize 20 \
         --circular-beam --fit-spec-order 2 \
         --name <name_prefix> \
         --ms ../visibility/*.ms
   ```

6. Convert image units from [Jy/beam] to [K] by using
   [`jybeam2k.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/oskar/jybeam2k.py):

   ```sh
   # Get the average beam size of the images among the frequency band
   $ beamsize *-????-image.fits
   $ for f in *-????-image.fits; do \
         jybeam2k.py -b <beamsize> ${f} ${f%.fits}K.fits; \
     done
   ```

7. Create image cube and crop by using
   [`fitscube.py`](https://github.com/liweitianux/atoolbox/blob/master/astro/fits/fitscube.py):

   ```sh
   $ fitscube.py create -z 154e6 -s 80e3 -u Hz \
         -o <prefix>_cube.fits -i images/*-imageK.fits
   $ fitscube.py crop -n 360 \
         -i <prefix>_cube.fits \
         -o <prefix>_n360_cube.fits
   ```
