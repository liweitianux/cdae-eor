#include "fftw_blitz.hpp"


namespace fftw_blitz{
  fftw_plan fftw_blitz_trait<double>::fft_plan_c2c(int rank, const int* n,
					      fftw_complex* in, 
					      fftw_complex* out,
					      int sign, unsigned int flags)
  {
    return fftw_plan_dft(rank,n,in,out,sign,flags);
  }

  fftw_plan fftw_blitz_trait<double>::fft_plan_r2c(int rank, const int* n,
				double* in, 
				fftw_complex* out,
				unsigned int flags)
  {
    return fftw_plan_dft_r2c(rank,n,in,out,flags);
  }


  fftw_plan fftw_blitz_trait<double>::fft_plan_c2r(int rank, const int* n,
				fftw_complex* in, 
				double* out,
				unsigned int flags)
  {
    return fftw_plan_dft_c2r(rank,n,in,out,flags);
  }

  fftwf_plan fftw_blitz_trait<float>::fft_plan_c2c(int rank, const int* n,
						  fftwf_complex* in, 
						  fftwf_complex* out,
						  int sign, unsigned int flags)
  {
    return fftwf_plan_dft(rank,n,in,out,sign,flags);
  }

  fftwf_plan fftw_blitz_trait<float>::fft_plan_r2c(int rank, const int* n,
						  float* in, 
						  fftwf_complex* out,
						  unsigned int flags)
  {
    return fftwf_plan_dft_r2c(rank,n,in,out,flags);
  }


  fftwf_plan fftw_blitz_trait<float>::fft_plan_c2r(int rank, const int* n,
						    fftwf_complex* in, 
						    float* out,
						    unsigned int flags)
  {
    return fftwf_plan_dft_c2r(rank,n,in,out,flags);
  }

  void fftw_blitz_trait<double>::fftw_init_threads()
  {
    ::fftw_init_threads();
  }

  void fftw_blitz_trait<float>::fftw_init_threads()
  {
    ::fftwf_init_threads();
  }

  void fftw_blitz_trait<double>::fftw_plan_with_nthreads(int nthreads)
  {
    ::fftw_plan_with_nthreads(nthreads);
  }
  
  void fftw_blitz_trait<float>::fftw_plan_with_nthreads(int nthreads)
  {
    ::fftwf_plan_with_nthreads(nthreads);
  }

  void fftw_blitz_trait<double>::fftw_execute(const fftw_plan plan)
  {
    ::fftw_execute(plan);
  }
  
  void fftw_blitz_trait<float>::fftw_execute(const fftwf_plan plan)
  {
    ::fftwf_execute(plan);
  }

  void fftw_blitz_trait<double>::fftw_destroy_plan(fftw_plan plan)
  {
    ::fftw_destroy_plan(plan);
  }

  void fftw_blitz_trait<float>::fftw_destroy_plan(fftwf_plan plan)
  {
    ::fftwf_destroy_plan(plan);
  }

  void fftw_blitz_trait<double>::fftw_cleanup_threads()
  {
    ::fftw_cleanup_threads();
  }

  void fftw_blitz_trait<float>::fftw_cleanup_threads()
  {
    ::fftwf_cleanup_threads();
  }
}
