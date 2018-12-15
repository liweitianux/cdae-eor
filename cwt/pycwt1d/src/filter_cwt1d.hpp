#ifndef FILTER_CWT1D
#define FILTER_CWT1D
#include "cwt1d.hpp"

namespace cwt1d
{
  template <typename T>
  blitz::Array<short,2> generate_mask(int signal_length,const blitz::Array<T,1>& scales,const wavelet_func<T>& wf,T delta_v=1,T step=.1)
  {
    //const T delta_v=.2;
    blitz::Array<T,1> fake_fg(signal_length);
    for(int i=0;i<signal_length;++i)
      {
	fake_fg(i)=i;
      }
    blitz::Array<std::complex<T>,2> wt_fake_fg(cwt(fake_fg,scales,wf));
    blitz::Array<T,2> mx(log10(abs(wt_fake_fg)));
#if 1
    cfitsfile ff;
    ff.create("a.fits");
    ff<<mx;
#endif
    T m2=-1e99;
    T m1=1e99;
    for(int i=0;i<mx.extent(0);++i)
      {
	for(int j=0;j<mx.extent(1);++j)
	  {
	    T v1=(T)mx(i,j);
	    if(isnan(v1))
	      {
		continue;
	      }
	    if(isinf(v1))
	      {
		continue;
	      }
	    m2=std::max(m2,v1);
	    m1=std::min(m1,v1);
	  }
      }
    int n_steps=(m2-m1)/step;
    blitz::Array<int,1> hist(n_steps+1);
    hist=0;
    for(int i=0;i<mx.extent(0);++i)
      {
	for(int j=0;j<mx.extent(1);++j)
	  {
	    int n=(mx(i,j)-m1)/step;
	    if(n<hist.extent(0)&&n>=0)
	      {
		hist(n)+=1;
	      }
	  }
      }
    int max_hist=0;
    int max_hist_idx=0;

#if 1
    std::ofstream ofs("hist.qdp");
    for(int i=0;i<n_steps;++i)
      {
	ofs<<i*step+m1<<"\t"<<hist(i)<<std::endl;
      }
#endif

    for(int i=0;i<n_steps;++i)
      {
	if(max_hist<hist(i))
	  {
	    max_hist=hist(i);
	    max_hist_idx=i;
	  }
      }
    T v=max_hist_idx*step+m1;
    blitz::Array<short,2> mask(mx.shape());
    mask=1;
    for(int j=0;j<mx.extent(1);++j)
      {
#if 1
	int i=0;
	for(i=mx.extent(0)-1;i>=0;--i)
	  {
	    if(mx(i,j)>v)
	      {
		break;
	      }
	  }
	for(;i>=0;--i)
	  {
	    if(mx(i,j)<v+delta_v)
	      {
		break;
	      }
	    mask(i,j)=0;
	  }
	for(i=0;i<mx.extent(0)-1;++i)
	  {
	    if(mx(i,j)>mx(i+1,j))
	      {
		mask(i,j)=0;
	      }
	    else
	      {
		break;
	      }
	  }

#else
	for(int i=0;i<mx.extent(0);++i)
	  {
	    if(mx(i,j)<v+delta_v)
	      {
		mask(i,j)=1;
	      }
	    else
	      {
		mask(i,j)=0;
	      }
	  }
#endif
      }
    return mask;
  }

  template <typename T>
  blitz::Array<T,1> smooth_pad(const blitz::Array<T,1>& x)
  {
    blitz::Array<T,1> result(x.extent(0)*3);
    T d1=(x(10)-x(0))/10.;
    T d2=(x(x.extent(0)-1)-x(x.extent(0)-11))/10.;
    for(int i=0;i<x.extent(0);++i)
      {
	result(i+x.extent(0))=x(i);
	result(i)=x(0)+(i-x.extent(0))*d1;
	result(i+2*x.extent(0))=x(x.extent(0)-1)+(i+1)*d2;
      }
    return result;
  }

  template<typename T>
  blitz::Array<T,1> smooth_unpad(const blitz::Array<T,1>& x)
  {
    blitz::Array<T,1> result(x.extent(0)/3);
    for(int i=0;i<result.extent(0);++i)
      {
	result(i)=x(result.extent(0)+i);
      }
    return result;
  }
}

#endif
