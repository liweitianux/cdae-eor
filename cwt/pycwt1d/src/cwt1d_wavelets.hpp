#ifndef CWT1D_WAVELETS
#define CWT1D_WAVELETS
#include "cwt1d.hpp"
#include <boost/math/special_functions/gamma.hpp>

namespace cwt1d
{
  template <typename T>
  class dog
    :public wavelet_func<T>
  {
  private:
    int m;
    T norm;
    T coi_factor;
    bool coi_calculated;
  public:
    dog()
      :m(2),coi_calculated(false)
    {
      norm=boost::math::tgamma(m+1/2.);
    }
    
    dog(int m1)
      :m(m1),coi_calculated(false)
    {
      norm=boost::math::tgamma(m+1/2.);
    }
    
    T do_cone_of_influence_factor()const
    {
      if(!coi_calculated)
	{
	  blitz::Array<T,1> signal(1024);
	  signal=0;
	  signal(signal.extent(0)/2)=1;
	  blitz::Array<T,1> scales(1);
	  scales(0)=1;
	  blitz::Array<std::complex<T>,2> signal_cwt(cwt(signal,scales,*this));
	  int max_idx=0;
	  for(int i=0;i<signal.extent(0)/2;++i)
	    {
	      if(abs(signal_cwt(0,i))>abs(signal_cwt(0,signal.extent(0)/2))/5)
		{
		  max_idx=i;
		  break;
		}
	    }
	  const_cast<T&>(coi_factor)=signal.extent(0)/2-max_idx;
	  const_cast<bool&>(coi_calculated)=true;
	}
      return coi_factor*.85;
    }
    
  private:
    std::complex<T> do_wavelet_f(T w,T s)const
    {
      const T ws=w*s;
      const T pi=4*std::atan(1);
      return -pow(std::complex<T>(0,1),T(m))*pow(ws,T(m))*exp(-(ws*ws)/2)/norm;
    }
  };
  
  template <typename T>
  class morlet
    :public wavelet_func<T>
  {
  private:
    T omega_0;
    T coi_factor;
    bool coi_calculated;
  public:
    morlet()
      :omega_0(1),coi_calculated(false)
    {}
    
    morlet(T w1)
      :omega_0(w1),coi_calculated(false)
    {}

    T do_cone_of_influence_factor()const
    {
      if(!coi_calculated)
	{
	  blitz::Array<T,1> signal(1024);
	  signal=0;
	  signal(signal.extent(0)/2)=1;
	  blitz::Array<T,1> scales(1);
	  scales(0)=8;
	  blitz::Array<std::complex<T>,2> signal_cwt(cwt(signal,scales,*this));
	  int max_idx=0;
	  for(int i=0;i<signal.extent(0)/2;++i)
	    {
	      if(abs(signal_cwt(0,i))>abs(signal_cwt(0,signal.extent(0)/2))/1e2)
		{
		  max_idx=i;
		  break;
		}
	    }
	  const_cast<T&>(coi_factor)=(signal.extent(0)/2-max_idx)/8.;
	  const_cast<bool&>(coi_calculated)=true;
	}
      return coi_factor*.85;
    }

    
  private:
    std::complex<T> do_wavelet_f(T w,T s)const
    {
      static const T pi=atan(1)*4;
      if(w<=0)
	{
	  return 0;
	}
      else
	{
	  return pow(T(pi),T(-.25))*exp(-pow(s*w-omega_0,2)/2);
	}
    }
  };

  template <typename T>
  class paul
    :public wavelet_func<T>
  {
  private:
    int m;
    T norm;
    T coi_factor;
    bool coi_calculated;

  public:
    paul()
      :m(4),coi_calculated(false)
    {
      norm=calc_norm();
    }
    
    paul(T m1)
      :m(m1),coi_calculated(false)
    {
      norm=calc_norm();
    }

    T do_cone_of_influence_factor()const
    {
      if(!coi_calculated)
	{
	  blitz::Array<T,1> signal(1024);
	  signal=0;
	  signal(signal.extent(0)/2)=1;
	  blitz::Array<T,1> scales(1);
	  scales(0)=1;
	  blitz::Array<std::complex<T>,2> signal_cwt(cwt(signal,scales,*this));
	  int max_idx=0;
	  for(int i=0;i<signal.extent(0)/2;++i)
	    {
	      if(abs(signal_cwt(0,i))>abs(signal_cwt(0,signal.extent(0)/2))/25)
		{
		  max_idx=i;
		  break;
		}
	    }
	  const_cast<T&>(coi_factor)=(signal.extent(0)/2-max_idx)/4.;
	  const_cast<bool&>(coi_calculated)=true;
	}
      return coi_factor;
    }

    
    T calc_norm()
    {
      return pow(T(2.),(T)m)/sqrt(m*factorial(2*m-1));
    }
    
    int factorial(int n)
    {
      if(n==0)
	{
	  return 1;
	}
      else
	{
	  return n*factorial(n-1);
	}
    }

  private:
    std::complex<T> do_wavelet_f(T w,T s)const
    {
      if(w<=0)
	{
	  return 0;
	}
      else
	{
	  return norm*pow(s*w,(T)m)*exp(-s*w);
	}
    }
  };
}

#endif
//EOF
