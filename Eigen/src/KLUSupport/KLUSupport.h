// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Kyle Macfarlan <kyle.macfarlan@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_KLUSUPPORT_H
#define EIGEN_KLUSUPPORT_H
#ifdef EIGEN_DUMP
#include <fstream>
#endif

namespace Eigen {

/* TODO extract L, extract U, compute det, etc... */

/** \ingroup KLUSupport_Module
  * \brief A sparse LU factorization and solver based on KLU
  *
  * This class allows to solve for A.X = B sparse linear problems via a LU factorization
  * using the KLU library. The sparse matrix A must be squared and full rank.
  * The vectors or matrices X and B can be either dense or sparse.
  *
  * \warning The input matrix A should be in a \b compressed and \b column-major form.
  * Otherwise an expensive copy will be made. You can call the inexpensive makeCompressed() to get a compressed matrix.
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  *
  * \implsparsesolverconcept
  *
  * \sa \ref TutorialSparseSolverConcept, class UmfPackLU, class SparseLU
  */


inline int klu_solve(klu_symbolic *Symbolic, klu_numeric *Numeric, Index ldim, Index nrhs, double B [ ], klu_common *Common, double) {
   return klu_solve(Symbolic, Numeric, internal::convert_index<int>(ldim), internal::convert_index<int>(nrhs), B, Common);
}

inline int klu_solve(klu_symbolic *Symbolic, klu_numeric *Numeric, Index ldim, Index nrhs, std::complex<double>B[], klu_common *Common, std::complex<double>) {
   return klu_z_solve(Symbolic, Numeric, internal::convert_index<int>(ldim), internal::convert_index<int>(nrhs), &numext::real_ref(B[0]), Common);
}

inline int klu_tsolve(klu_symbolic *Symbolic, klu_numeric *Numeric, Index ldim, Index nrhs, double B[], klu_common *Common, double) {
   return klu_tsolve(Symbolic, Numeric, internal::convert_index<int>(ldim), internal::convert_index<int>(nrhs), B, Common);
}

inline int klu_tsolve(klu_symbolic *Symbolic, klu_numeric *Numeric, Index ldim, Index nrhs, std::complex<double>B[], klu_common *Common, std::complex<double>) {
   return klu_z_tsolve(Symbolic, Numeric, internal::convert_index<int>(ldim), internal::convert_index<int>(nrhs), &numext::real_ref(B[0]), 0, Common);
}

inline klu_numeric* klu_factor(int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_common *Common, double) {
   return klu_factor(Ap, Ai, Ax, Symbolic, Common);
}

inline klu_numeric* klu_factor(int Ap[], int Ai[], std::complex<double> Ax[], klu_symbolic *Symbolic, klu_common *Common, std::complex<double>) {
   return klu_z_factor(Ap, Ai, &numext::real_ref(Ax[0]), Symbolic, Common);
}


template<typename _MatrixType>
class KLU : public SparseSolverBase<KLU<_MatrixType> >
{
  protected:
    typedef SparseSolverBase<KLU<_MatrixType> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    typedef Matrix<int, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef SparseMatrix<Scalar> LUMatrixType;
    typedef SparseMatrix<Scalar,ColMajor,int> KLUMatrixType;
    typedef Ref<const KLUMatrixType, StandardCompressedFormat> KLUMatrixRef;

    typedef unsigned int UInt;
    std::vector<std::pair<UInt, UInt>> changedEntries;
    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:

    KLU()
      : m_dummy(0,0), mp_matrix(m_dummy)
    {
      init();
    }

    template<typename InputMatrixType>
    explicit KLU(const InputMatrixType& matrix)
      : mp_matrix(matrix)
    {
      init();
      compute(matrix);
    }

    ~KLU()
    {
      if(m_symbolic) klu_free_symbolic(&m_symbolic,&m_common);
      if(m_numeric)  klu_free_numeric(&m_numeric,&m_common);
    }

    EIGEN_CONSTEXPR inline Index rows() const EIGEN_NOEXCEPT { return mp_matrix.rows(); }
    EIGEN_CONSTEXPR inline Index cols() const EIGEN_NOEXCEPT { return mp_matrix.cols(); }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was successful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }
#if 0 // not implemented yet
    inline const LUMatrixType& matrixL() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_l;
    }

    inline const LUMatrixType& matrixU() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_u;
    }

    inline const IntColVectorType& permutationP() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_p;
    }

    inline const IntRowVectorType& permutationQ() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_q;
    }
#endif
    /** Computes the sparse Cholesky decomposition of \a matrix
     *  Note that the matrix should be column-major, and in compressed format for best performance.
     *  \sa SparseMatrix::makeCompressed().
     */
    template<typename InputMatrixType>
    void compute(const InputMatrixType& matrix)
    {
      if(m_symbolic) klu_free_symbolic(&m_symbolic, &m_common);
      if(m_numeric)  klu_free_numeric(&m_numeric, &m_common);
      grab(matrix.derived());
      analyzePattern_impl();
      factorize_impl();
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize(), compute()
      */
    template<typename InputMatrixType>
    void analyzePattern(const InputMatrixType& matrix)
    {
      if(m_symbolic) klu_free_symbolic(&m_symbolic, &m_common);
      if(m_numeric)  klu_free_numeric(&m_numeric, &m_common);

      grab(matrix.derived());

      analyzePattern_impl();
    }
   
    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize(), compute()
      */
    template<typename InputMatrixType, typename ListType>
    void analyzePatternPartial(const InputMatrixType& matrix, const ListType& list, const int mode)
    {
      if(m_symbolic) klu_free_symbolic(&m_symbolic, &m_common);
      if(m_numeric)  klu_free_numeric(&m_numeric, &m_common);

      m_mode = mode;
      grab(matrix.derived());
      this->changedEntries = list;
      analyzePattern_impl();
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize(), compute()
      */
    template<typename InputMatrixType, typename ListType>
    void analyzePattern(const InputMatrixType& matrix, const ListType& list)
    {
      if(m_symbolic) klu_free_symbolic(&m_symbolic, &m_common);
      if(m_numeric)  klu_free_numeric(&m_numeric, &m_common);

      grab(matrix.derived());

      analyzePattern_impl();
    }
#ifdef EIGEN_DUMP
    template<typename InputMatrixType>
    void printMTX(const InputMatrixType& matrix, int counter)
    {
	int i, j;
	int n = internal::convert_index<int>(matrix.rows());

 	StorageIndex* Ap = const_cast<StorageIndex*>(matrix.outerIndexPtr());
	StorageIndex* Ai = const_cast<StorageIndex*>(matrix.innerIndexPtr()); 
	Scalar* Ax = const_cast<Scalar*>(mp_matrix.valuePtr());
	int nz = matrix.nonZeros();

	std::ofstream ofs;
	char strA[32];
	char counterstring[32];
	sprintf(counterstring, "%d", counter);
	strcpy(strA, "A");
	strcat(strA, counterstring);
	strcat(strA, ".mtx");
	ofs.open(strA);
	ofs << "%%MatrixMarket matrix coordinate real general" << std::endl;
	ofs << n << " " << n << " " << nz << std::endl;
	for(i = 0 ; i < n ; i++)
	{
		for(j = Ap[i] ; j < Ap[i+1] ; j++)
		{
			ofs << i+1 << " " << Ai[j]+1 << " " << Ax[j] << std::endl;
		}
	}
	ofs.close();
    }
#endif


    /** Provides access to the control settings array used by KLU.
      *
      * See KLU documentation for details.
      */
    inline const klu_common& kluCommon() const
    {
      return m_common;
    }

    /** Provides access to the control settings array used by UmfPack.
      *
      * If this array contains NaN's, the default values are used.
      *
      * See KLU documentation for details.
      */
    inline klu_common& kluCommon()
    {
      return m_common;
    }

    /** Performs a numeric decomposition of \a matrix and computes factorization path
     *
     * The given matrix must has the same sparcity than the matrix on which the
     * pattern anylysis has been performed.
     *
     * \sa analyzePattern(), compute()
     */
    template <typename InputMatrixType, typename ListType>
    void factorize_partial(const InputMatrixType &matrix, const ListType& variableList) {
      eigen_assert(m_analysisIsOk && "KLU: you must first call analyzePattern()");
      
      /* m_dump = doDump;
       * m_common.dump = m_dump;
       */
#ifdef EIGEN_DUMP
      m_limit = 2;
      char *limit = getenv("EIGEN_MATRIX_LIMIT");
      if(limit != NULL)
      {
	 m_limit = atoi(limit);
	 if(m_limit <= 0)
	 {
		 m_limit = 2;
	 }
      }
#endif
      grab(matrix.derived());
      this->changedEntries = variableList;
      factorize_with_path_impl();
    }

    /** Placeholder function. Should be deleted / improved later
     *
     * \sa analyzePattern(), compute()
     */
    template <typename InputMatrixType>
    void factorize_partial(const InputMatrixType &matrix) {
      eigen_assert(m_analysisIsOk && "KLU: you must first call analyzePattern()");
      
      /*
       * m_dump = 0;
       * m_common.dump = m_dump;
      */
      grab(matrix.derived());
      factorize_with_path_impl();
    }

    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the pattern anylysis has been performed.
      *
      * \sa analyzePattern(), compute()
      */
    template<typename InputMatrixType>
    void factorize(const InputMatrixType& matrix)
    {
      eigen_assert(m_analysisIsOk && "KLU: you must first call analyzePattern()");
      if(m_numeric)
      {
        klu_free_numeric(&m_numeric,&m_common);
      }

      grab(matrix.derived());
      if(m_symbolic->nz != mp_matrix.nonZeros())
      {
        analyzePattern_impl();
      }
      factorize_impl();
    }

    /** Performs a numeric re-decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the pattern anylysis has been performed.
      * The pivoting values are chosen the same
      *
      * \sa analyzePattern(), compute()
      */
    template<typename InputMatrixType>
    void refactorize(const InputMatrixType& matrix)
    {
      eigen_assert(m_analysisIsOk && "KLU: you must first call analyzePattern()");
      eigen_assert(m_factorizationIsOk && "KLU: you must first call factorize()");
      /**/

      grab(matrix.derived());
      /*temporary workaround for a problem in DPsim (?)*/
      if(m_symbolic->nz != mp_matrix.nonZeros()){
        if(m_numeric)
          klu_free_numeric(&m_numeric,&m_common);
        analyzePattern_impl();
        factorize_impl();
      } else {
        refactorize_impl();
      }
    }

    /** Performs a numeric partial re-decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the pattern anylysis has been performed.
      * The pivoting values are chosen the same
      * The re-decomposition is performed at a minimal subset of columns
      *
      * \sa analyzePattern(), compute()
      */
    template<typename InputMatrixType>
    void refactorize_partial(const InputMatrixType& matrix)
    {
      eigen_assert(m_analysisIsOk && "KLU: you must first call analyzePattern()");
      eigen_assert(m_factorizationIsOk && "KLU: you must first call factorize_partial()");
      eigen_assert(m_partial_is_ok && "KLU: factorization path is not OK");
      /* TODO: eigen_assert(m_factorizationPathIsOk ... ) */
      /**/

      grab(matrix.derived());
      // temporary workaround for a problem in DPsim
      if(m_symbolic->nz != mp_matrix.nonZeros()){
        if(m_numeric)
          klu_free_numeric(&m_numeric,&m_common);
        analyzePattern_impl();
        factorize_with_path_impl();
      } else {
        refactorize_partial_impl();
      }
    }

    /** \internal */
    template<typename BDerived,typename XDerived>
    bool _solve_impl(const MatrixBase<BDerived> &b, MatrixBase<XDerived> &x) const;

#if 0 // not implemented yet
    Scalar determinant() const;

    void extractData() const;
#endif

  protected:

    void init()
    {
      m_info                  = InvalidInput;
      m_isInitialized         = false;
      m_numeric               = 0;
      m_symbolic              = 0;
      m_extractedDataAreDirty = true;

      klu_defaults(&m_common);

      char* variable = getenv("KLU_SCALING");
      m_scale = 0;
      if(variable!=NULL)
      {
        m_scale = atoi(variable);
        if(m_scale > 2)
        {
          m_scale = 0;
        }
      }
      variable = getenv("KLU_BTF");
      m_btf = 1;
      if(variable!=NULL)
      {
        m_btf = atoi(variable);
        if(m_btf<0||m_btf>1)
        {
          m_btf = 1;
        }
      }
      m_common.btf = m_btf;
      m_common.scale = m_scale;
    }

    void analyzePattern_impl()
    {
      m_info = InvalidInput;
      m_analysisIsOk = false;
      m_factorizationIsOk = false;
      m_refactorizationIsOk = false;
      m_partial_refactorizationIsOk = false;
      m_partial_is_ok = false;

      const int n = internal::convert_index<int>(mp_matrix.rows());

      if(m_mode == KLU_AMD_BRA_RR || m_mode == KLU_AMD_NV_FP)
      {
        int varying_entries = changedEntries.size();
        int *varying_columns = (int*)calloc(sizeof(int), varying_entries);
        int *varying_rows = (int*)calloc(sizeof(int), varying_entries);
        int counter = 0;
        for(std::pair<UInt, UInt> i : changedEntries){
          varying_rows[counter] = i.first;
          varying_columns[counter] = i.second;
          counter++;
        }

        m_symbolic = klu_analyze_partial(n,
                                      const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()),
                                      varying_columns, varying_rows, varying_entries, m_mode, &m_common);

        free(varying_rows);
        free(varying_columns);
      }
      else
      {
        m_symbolic = klu_analyze(n,
                                     const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()),
                                    &m_common);
      }
      if (m_symbolic) {
         m_isInitialized = true;
         m_info = Success;
         m_analysisIsOk = true;
         m_extractedDataAreDirty = true;
      }
    }

    void factorize_impl()
    {

      m_numeric = klu_factor(const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()), const_cast<Scalar*>(mp_matrix.valuePtr()),
                                    m_symbolic, &m_common, Scalar());


      m_info = m_numeric ? Success : NumericalIssue;
      m_factorizationIsOk = m_numeric ? 1 : 0;
      m_extractedDataAreDirty = true;
    }

    void factorize_with_path_impl() {
      StorageIndex* Ap = const_cast<StorageIndex*>(mp_matrix.outerIndexPtr());
      StorageIndex* Ai = const_cast<StorageIndex*>(mp_matrix.innerIndexPtr());
      Scalar* Ax = const_cast<Scalar*>(mp_matrix.valuePtr());
      m_numeric = klu_factor(Ap, Ai, Ax, m_symbolic, &m_common, Scalar());

      int varying_entries = changedEntries.size();
      int *varying_columns = (int*)calloc(sizeof(int), varying_entries);
      int *varying_rows = (int*)calloc(sizeof(int), varying_entries);
      int counter = 0;
      for(std::pair<UInt, UInt> i : changedEntries){
        varying_rows[counter] = i.first;
        varying_columns[counter] = i.second;
        counter++;
      }

      if(m_mode == KLU_AMD_FP || m_mode == KLU_AMD_NV_FP)
      {
        m_partial_is_ok = klu_compute_path(m_symbolic, m_numeric, &m_common, Ap, Ai, varying_columns, varying_rows, varying_entries);
      }
      else if(m_mode == KLU_AMD_RR || m_mode == KLU_AMD_BRA_RR)
      {
        m_partial_is_ok = klu_determine_start(m_symbolic, m_numeric, &m_common, Ap, Ai, varying_columns, varying_rows, varying_entries);
      }
      m_info = m_numeric ? Success : NumericalIssue;
      m_factorizationIsOk = m_numeric ? 1 : 0;
      m_extractedDataAreDirty = true;

      free(varying_columns);
      free(varying_rows);
    }

    void refactorize_impl()
    {
      int m_refact = klu_refactor(const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()), const_cast<Scalar*>(mp_matrix.valuePtr()),
                                    m_symbolic, m_numeric, &m_common);

      m_info = m_refact ? Success : NumericalIssue;
      m_refactorizationIsOk = m_refact ? 1 : 0;
      m_extractedDataAreDirty = true;
    }

    void refactorize_partial_impl()
    {
#ifdef EIGEN_DUMP
      static int counter = 1;
      if(counter == 1 || counter == m_limit)
      {
	      printMTX(mp_matrix, counter);
      }
      counter++;
#endif
      int m_partial_refact = 0;
      if(m_mode == KLU_AMD_FP || m_mode == KLU_AMD_NV_FP)
      {
        m_partial_refact = klu_partial_factorization_path(const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()), const_cast<Scalar*>(mp_matrix.valuePtr()),
                                    m_symbolic, m_numeric, &m_common);
      }
      else if(m_mode == KLU_AMD_RR || m_mode == KLU_AMD_BRA_RR)
      {
        m_partial_refact = klu_partial_refactorization_restart(const_cast<StorageIndex*>(mp_matrix.outerIndexPtr()), const_cast<StorageIndex*>(mp_matrix.innerIndexPtr()), const_cast<Scalar*>(mp_matrix.valuePtr()),
                                    m_symbolic, m_numeric, &m_common);
      }
      if(m_common.status == KLU_PIVOT_FAULT){
        /* pivot became too small => fully factorize again */
        factorize_impl();
      }
      m_info = m_partial_refact ? Success : NumericalIssue;
      m_partial_refactorizationIsOk = m_partial_refact ? 1 : 0;
      m_extractedDataAreDirty = true;
    }

    template<typename MatrixDerived>
    void grab(const EigenBase<MatrixDerived> &A)
    {
      mp_matrix.~KLUMatrixRef();
      ::new (&mp_matrix) KLUMatrixRef(A.derived());
    }

    void grab(const KLUMatrixRef &A)
    {
      if(&(A.derived()) != &mp_matrix)
      {
        mp_matrix.~KLUMatrixRef();
        ::new (&mp_matrix) KLUMatrixRef(A);
      }
    }

    // cached data to reduce reallocation, etc.
#if 0 // not implemented yet
    mutable LUMatrixType m_l;
    mutable LUMatrixType m_u;
    mutable IntColVectorType m_p;
    mutable IntRowVectorType m_q;
#endif

    KLUMatrixType m_dummy;
    KLUMatrixRef mp_matrix;

    klu_numeric* m_numeric;
    klu_symbolic* m_symbolic;
    klu_common m_common;
    mutable ComputationInfo m_info;
    int m_factorizationIsOk;
    int m_refactorizationIsOk;
    int m_partial_refactorizationIsOk;
    int m_analysisIsOk;
    int m_partial_is_ok;
    mutable bool m_extractedDataAreDirty;
    int m_dump;
    int m_scale;
    int m_limit;
    int m_mode;
    int m_btf;

  private:
    KLU(const KLU& ) { }
};

#if 0 // not implemented yet
template<typename MatrixType>
void KLU<MatrixType>::extractData() const
{
  if (m_extractedDataAreDirty)
  {
     eigen_assert(false && "KLU: extractData Not Yet Implemented");

    // get size of the data
    int lnz, unz, rows, cols, nz_udiag;
    umfpack_get_lunz(&lnz, &unz, &rows, &cols, &nz_udiag, m_numeric, Scalar());

    // allocate data
    m_l.resize(rows,(std::min)(rows,cols));
    m_l.resizeNonZeros(lnz);

    m_u.resize((std::min)(rows,cols),cols);
    m_u.resizeNonZeros(unz);

    m_p.resize(rows);
    m_q.resize(cols);

    // extract
    umfpack_get_numeric(m_l.outerIndexPtr(), m_l.innerIndexPtr(), m_l.valuePtr(),
                        m_u.outerIndexPtr(), m_u.innerIndexPtr(), m_u.valuePtr(),
                        m_p.data(), m_q.data(), 0, 0, 0, m_numeric);

    m_extractedDataAreDirty = false;
  }
}

template<typename MatrixType>
typename KLU<MatrixType>::Scalar KLU<MatrixType>::determinant() const
{
  eigen_assert(false && "KLU: extractData Not Yet Implemented");
  return Scalar();
}
#endif

template<typename MatrixType>
template<typename BDerived,typename XDerived>
bool KLU<MatrixType>::_solve_impl(const MatrixBase<BDerived> &b, MatrixBase<XDerived> &x) const
{
  Index rhsCols = b.cols();
  EIGEN_STATIC_ASSERT((XDerived::Flags&RowMajorBit)==0, THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  eigen_assert(m_factorizationIsOk && "The decomposition is not in a valid state for solving, you must first call either compute() or analyzePattern()/factorize()");

  x = b;
  int info = klu_solve(m_symbolic, m_numeric, b.rows(), rhsCols, x.const_cast_derived().data(), const_cast<klu_common*>(&m_common), Scalar());

  m_info = info!=0 ? Success : NumericalIssue;
  return true;
}

} // end namespace Eigen

#endif // EIGEN_KLUSUPPORT_H
