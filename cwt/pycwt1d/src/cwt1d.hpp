#ifndef CWT1D
#define CWT1D
#include <complex>
#include <fftw_blitz.hpp>

namespace cwt1d
{
  template <typename T>
  class wavelet_func
  {
  public:
    //w is omega
    //s is scale
    std::complex<T> wavelet_f(T w,T s)const
    {
      return do_wavelet_f(w,s);
    }

    T cone_of_influence_factor()const
    {
      return do_cone_of_influence_factor();
    }
  private:
    virtual std::complex<T> do_wavelet_f(T w,T s)const=0;
    virtual T do_cone_of_influence_factor()const
    {
      return 1;
    }
  };

  template <typename T>
  blitz::Array<std::complex<T>,2> cwt(const blitz::Array<T,1>& x,const blitz::Array<T,1>& s,const wavelet_func<T>& wf)
  {
    const T pi=4*std::atan(1);
    blitz::Array<std::complex<T>,2> result(s.extent(0),x.extent(0));
    blitz::Array<std::complex<T>,1> cx(x.shape());
    for(int i=0;i<x.extent(0);++i)
      {
	cx(i)=x(i);
      }
    blitz::Array<std::complex<T>,1> buff(fftw_blitz::fft_c2c(cx,FFTW_FORWARD));
    blitz::Array<std::complex<T>,1> buff1(x.shape());
    for(int i=0;i<s.extent(0);++i)
      {
	for(int k=0;k<x.extent(0);++k)
	  {
	    const T omega_k=2.*pi/T(x.extent(0))*(2*k<=x.extent(0)?k:k-x.extent(0));

	    buff1(k)=buff(k)*std::conj(std::sqrt(std::abs(s(i)))*wf.wavelet_f(omega_k,s(i)));
	  }
	buff1=fftw_blitz::fft_c2c(buff1,FFTW_BACKWARD)/T(2.)/T(pi)/T(buff1.extent(0));
	for(int k=0;k<x.extent(0);++k)
	  {
	    result(i,k)=buff1(k);
	  }
      }
    return result;
  }

  template <typename T>
  blitz::Array<T,1> icwt(const blitz::Array<std::complex<T>,2>& x,const blitz::Array<T,1>& s,const wavelet_func<T>& wf)
  {
    const T pi=4*std::atan(1);
    assert(s.extent(0)==x.extent(0));
    blitz::Array<std::complex<T>,1> xbuff(x.extent(1));
    blitz::Array<std::complex<T>,1> wbuff(x.extent(1));
    blitz::Array<T,2> tmp_mx(x.shape());
    blitz::Array<T,1> result(x.extent(1));
    //integrate time:
    for(int i=0;i<x.extent(0);++i)
      {
	for(int j=0;j<x.extent(1);++j)
	  {
	    const T omega_j=2.*pi/T(x.extent(1))*(2*j<=x.extent(1)?j:j-x.extent(1));
	    xbuff(j)=x(i,j);
	    wbuff(j)=std::conj(std::sqrt(std::abs(s(i)))*wf.wavelet_f(omega_j,s(i)));
	  }
	xbuff=fftw_blitz::fft_c2c(xbuff);
	xbuff*=wbuff;
	xbuff=fftw_blitz::fft_c2c(xbuff,FFTW_BACKWARD)/s(i)/s(i);
	for(int j=0;j<x.extent(1);++j)
	  {
	    tmp_mx(i,j)=xbuff(j).real();///s(i)/s(i);
	  }
      }

    
    result=0;
    for(int i=0;i<x.extent(0)-1;++i)
      {
	for(int j=0;j<x.extent(1);++j)
	  {
	    result(j)+=(tmp_mx(i,j)+tmp_mx(i+1,j))/2.*(s(i+1)-s(i));
	  }
      }
    
    return result;
  }
  
  template <typename T>
  T calc_norm(size_t signal_length,const blitz::Array<T,1>& s,const wavelet_func<T>& wf)
  {
    blitz::Array<T,1> dummy_signal(signal_length);
    T l=s(s.extent(0)/2);
    l=l<1?1:l;
    T pi=std::atan(T(1));
    for(size_t i=0;i<signal_length;++i)
      {
	dummy_signal(i)=std::cos(i/l*2*pi);
      }
    dummy_signal-=blitz::mean(dummy_signal);
    blitz::Array<T,1> recovered_signal(icwt(cwt(dummy_signal,s,wf),s,wf));
    recovered_signal-=blitz::mean(recovered_signal);
    //std::cerr<<blitz::mean(dummy_signal*dummy_signal)<<std::endl;    
    //std::cerr<<blitz::mean(recovered_signal*recovered_signal)<<std::endl;
    return std::sqrt(blitz::mean(recovered_signal*recovered_signal)/
		     blitz::mean(dummy_signal*dummy_signal));

  }
}

#endif
