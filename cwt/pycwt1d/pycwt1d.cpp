#include <iostream>
#include <vector>
#include <boost/ref.hpp>
#include <boost/utility.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/def.hpp>
#include <boost/python/pure_virtual.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/operators.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/list.hpp>
#include <exception>
#define private public

using namespace boost;
using namespace boost::python;
using namespace boost::python::numpy;

#include <cwt1d_wavelets.hpp>
using namespace std;
using namespace cwt1d;

typedef dog<double> pydog;
typedef morlet<double> pymorlet;
typedef paul<double> pypaul;
typedef wavelet_func<double> wf;

namespace
{
  class initializer{
  public:
    initializer()
    {
      //boost::python::numeric::array::set_module_and_type("numpy","ndarray");
      Py_Initialize();
      numpy::initialize();
    }
  }_init;
}

boost::python::numpy::ndarray pycwt(const boost::python::numpy::ndarray& x,const boost::python::numpy::ndarray& s,const wavelet_func<double>& wf)
{
  boost::python::object shape(x.attr("shape"));
  int ndim=extract<int>(shape.attr("__len__")());
  if(ndim!=1)
    {
      throw std::exception();
    }
  int ndata=extract<int>(shape[0]);
  
  blitz::Array<double,1> x1(ndata);
  for(int i=0;i<ndata;++i)
    {
      x1(i)=extract<double>(x[i]);
    }
  shape=s.attr("shape");
  ndim=extract<int>(shape.attr("__len__")());
  if(ndim!=1)
    {
      throw std::exception();
    }
  int nscales=extract<int>(shape[0]);
  blitz::Array<double,1> s1(nscales);

  for(int i=0;i<nscales;++i)
    {
      s1(i)=extract<double>(s[i]);
    }

  blitz::Array<complex<double>,2> y(cwt(x1,s1,wf));
  boost::python::list l;
  for(int i=0;i<nscales;++i)
    {
      boost::python::list l1;
      for(int j=0;j<ndata;++j)
	{
	  l1.append(y(i,j));
	}
      l.append(l1);
    }
  
  return boost::python::numpy::array(l);

}

boost::python::numpy::ndarray pyicwt(const boost::python::numpy::ndarray& x,const boost::python::numpy::ndarray& s,const wavelet_func<double>& wf)
{
  boost::python::object shape(x.attr("shape"));
  int ndim=extract<int>(shape.attr("__len__")());
  if(ndim!=2)
    {
      throw std::exception();
    }
  int ndata=extract<int>(shape[1]);
  int nscales=extract<int>(shape[0]);
  
  blitz::Array<complex<double>,2> x1(nscales,ndata);
  for(int i=0;i<nscales;++i)
    {
      for(int j=0;j<ndata;++j)
	{
	  x1(i,j)=extract<complex<double> >(x[boost::python::make_tuple(i,j)]);
	}
    }
  shape=s.attr("shape");
  ndim=extract<int>(shape.attr("__len__")());
  if(ndim!=1)
    {
      throw std::exception();
    }
  if(nscales!=extract<int>(shape[0]))
    {
      throw std::exception();
    }
  blitz::Array<double,1> s1(nscales);

  for(int i=0;i<nscales;++i)
    {
      s1(i)=extract<double>(s[i]);
    }
  blitz::Array<double,1> result1(icwt(x1,s1,wf));
  boost::python::list l;
  for(int i=0;i<ndata;++i)
    {
      l.append(result1(i));
    }
  return boost::python::numpy::array(l);
  //return result;
}

boost::python::numpy::ndarray generate_log_scales(double min_scale,double max_scale,int num_scales)
{
  boost::python::list l;
  double lmin_scale=log(min_scale);
  double lmax_scale=log(max_scale);
  for(int i=0;i<num_scales;++i)
    {
      double s=exp(lmin_scale+(lmax_scale-lmin_scale)/(num_scales-1)*i);
      l.append(s);
    }
  return boost::python::numpy::array(l);
  //  return result;
}

double pycalc_norm(int dl,const boost::python::numpy::ndarray& s,const wavelet_func<double>& wf)
{
  boost::python::object shape(s.attr("shape"));
  int ndim=extract<int>(shape.attr("__len__")());
  if(ndim!=1)
    {
      throw std::exception();
    }
  int nscales=extract<int>(shape[0]);
  blitz::Array<double,1> s1(nscales);
  for(int i=0;i<nscales;++i)
    {
      s1(i)=extract<double>(s[i]);
    }
  return calc_norm(dl,s1,wf);
}

BOOST_PYTHON_MODULE(cwtcore)
{
  class_<wf,boost::noncopyable>("wf",no_init)
    .def("wavelet_f",&wf::wavelet_f);
  
  class_<pydog,bases<wf> >("dog")
    .def(init<>())
    .def(init<int>());

  class_<pymorlet,bases<wf> >("morlet")
    .def(init<>())
    .def(init<double>());
  
  class_<pypaul,bases<wf> >("paul")
    .def(init<>())
    .def(init<double>());


  def("cwt",pycwt);
  def("icwt",pyicwt);
  def("calc_norm",pycalc_norm);
  def("generate_log_scales",generate_log_scales);
}
