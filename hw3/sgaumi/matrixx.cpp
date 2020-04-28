#include <iostream>
//#include <lapacke.h>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

class Matrix {

public:

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix & operator=(std::vector<double> const & vec)
    {
        if (size() != vec.size())
        {
            throw std::out_of_range("number of elements mismatch");
        }

        size_t k = 0;
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = vec[k];
                ++k;
            }
        }

        return *this;
    }

    Matrix(Matrix const & other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    Matrix & operator=(Matrix const & other)
    {
        if (this == &other) { return *this; }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
        }
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
        return *this;
    }

    Matrix(Matrix && other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
    }

    Matrix & operator=(Matrix && other)
    {
        if (this == &other) { return *this; }
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
        return *this;
    }

    ~Matrix()
    {
        reset_buffer(0, 0);
    }

    double   operator() (size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[index(row, col)]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
	double *buff() const { return m_buffer; }
	double *buff() { return m_buffer; }

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    std::vector<double> buffer_vector() const { return std::vector<double>(m_buffer, m_buffer+size()); }

private:

    size_t index(size_t row, size_t col) const
    {
        return row + col * m_nrow;
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement]; }
        else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double * m_buffer = nullptr;

};

Matrix multiply_naive(Matrix const& m1, Matrix const& m2){
	size_t n1=m1.nrow();
	size_t n2=m2.ncol();
	Matrix res = Matrix(n1,n2);
	for (size_t i=0;i<n1;i++){
		for (size_t j=0;j<n2;j++){
			double s=0;
			for(size_t k=0;k<m1.ncol();k++){
				s=s+m1(i,k)*m2(k,j);
			}
			std::cout<<"i"<<i<<"j"<<j<<"s"<<s<<std::endl;
			res(i,j)=s;
			std::cout<<"resij"<<res(i,j)<<std::endl;
		}
	}
	return res;
	
}

/*Matrix multiply_mkl(Matrix const& m1, Matrix const& m2)
{
    size_t n1 = a.nrow();
    size_t k = a.ncol();
    size_t n2 = b.ncol();
    Matrix m = Matrix(n1, n2);

	cblas_dgemm(CblasRowMajor, //CblasRowMajor=101
	CblasNoTrans, //CblasNoTrans=111
	CblasNoTrans, //CblasNoTrans=111
    n1, n2, k, 1,
    m1.buff(), k,
    m2.buff(), n2,
    0, m.buff(), n2);

    return m;
}*/

PYBIND11_MODULE(matrixx, m){
    m.def("multiply_naive", &multiply_naive);
    //m.def("multiply_mkl", &multiply_mkl);

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init([](size_t r, size_t c) {return new Matrix(r, c);}))
        .def(py::init<Matrix&>())
        .def(py::init<std::vector<std::vector<double>>&>())
        .def_property("nrow", &Matrix::nrow, nullptr)
        .def_property("ncol", &Matrix::ncol, nullptr)
        //.def("__eq__", &Matrix::operator==)
        //.def("__repr__", &Matrix::reprString)
        .def_buffer([](Matrix &m) -> py::buffer_info{
            return py::buffer_info(
                m.buff(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                { m.nrow(), m.ncol() },
                { sizeof(double)*m.ncol(), sizeof(double) }
            );
        })
        /*.def("__setitem__", [](Matrix &m, std::array<double, 2> tp, double val){
            m.set_data(tp[0], tp[1], val);
        })
        .def("__getitem__", [](Matrix &m, std::array<double, 2> tp){
            return m.get_data(tp[0], tp[1]);
        })*/
        ;
}
