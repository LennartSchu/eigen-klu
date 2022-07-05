// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Lennart Schumacher
// <lennart.schumacher@eonerc.rwth-aachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NICSLUSUPPORT_H
#define EIGEN_NICSLUSUPPORT_H
#ifdef EIGEN_DUMP
#include <fstream>
#endif

namespace Eigen {

/* TODO extract L, extract U, compute det, etc... */

/** \ingroup NICSLUSupport_Module
 * \brief A sparse LU factorization and solver based on NICSLU
 *
 * This class allows to solve for A.X = B sparse linear problems via a LU
 * factorization using the NICSLU library. The sparse matrix A must be squared
 * and full rank. The vectors or matrices X and B can be either dense or sparse.
 *
 * \warning The input matrix A should be in a \b compressed and \b row-major
 * form. Otherwise an expensive copy will be made. You can call the inexpensive
 * makeCompressed() to get a compressed matrix. \tparam _MatrixType the type of
 * the sparse matrix A, it must be a SparseMatrix<>
 *
 * \implsparsesolverconcept
 *
 * \sa \ref TutorialSparseSolverConcept, class NicsLU
 */

inline int nicslu_solve(SNicsLU *nicslu, real__t *rhs) {
  return NicsLU_Solve(nicslu, rhs);
}

inline int nicslu_tsolve(SNicsLU *nicslu, real__t *rhs) {
  return NicsLU_Solve(nicslu, rhs);
}

inline int nicslu_factor(SNicsLU *nicslu) { return NicsLU_Factorize(nicslu); }

template <typename _MatrixType>
class NICSLU : public SparseSolverBase<NICSLU<_MatrixType>> {
protected:
  typedef SparseSolverBase<NICSLU<_MatrixType>> Base;
  using Base::m_isInitialized;

public:
  using Base::_solve_impl;
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Matrix<Scalar, Dynamic, 1> Vector;

  typedef SparseMatrix<Scalar, ColMajor, int> NICSLUMatrixType;
  typedef Matrix<int, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
  typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;

  /*typedef Matrix<int, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
  typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
  typedef SparseMatrix<Scalar,RowMajor,int> NICSLUMatrixType;*/

  typedef SparseMatrix<Scalar> LUMatrixType;
  typedef Ref<const NICSLUMatrixType, StandardCompressedFormat> NICSLUMatrixRef;

  typedef unsigned int UInt;
  std::vector<std::pair<UInt, UInt>> changedEntries;
  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime
  };

public:
  NICSLU() : m_dummy(0, 0), mp_matrix(m_dummy) { init(); }

  /*template<typename InputMatrixType>
  explicit NICSLU(const InputMatrixType& matrix) :
    mp_matrix(m_dummy)
  {
  }*/

  /** \returns the LU decomposition matrix: the upper-triangular part is U, the
   * unit-lower-triangular part is L (at least for square matrices; in the
   * non-square case, special care is needed, see the documentation of class
   * FullPivLU).
   *
   * \sa matrixL(), matrixU()
   */
  inline const MatrixType &matrixLU() const {
    eigen_assert(m_isInitialized && "NICSLU is not initialized.");
    return nicslu->lu_array;
  }

  NICSLU(const NICSLU &) : mp_matrix(m_dummy) {}

  template <typename InputMatrixType>
  NICSLU &operator=(const NICSLU<InputMatrixType> &other) {}

  template <typename InputMatrixType>
  explicit NICSLU(const InputMatrixType &matrix) : mp_matrix(m_dummy) {
    /*mp_matrix = matrix;
    init();
    compute(matrix);*/
  }

  ~NICSLU() {
    if (m_symbolic) {
      NicsLU_Destroy(nicslu);
      m_isInitialized = false;
      nicslu = NULL;
      //nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
      m_symbolic = 0;
      m_numeric = 0;
    }
  }

  inline Index rows() const { return mp_matrix.rows(); }
  inline Index cols() const { return mp_matrix.cols(); }

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful,
   *          \c NumericalIssue if the matrix.appears to be negative.
   */
  ComputationInfo info() const {
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
   *  Note that the matrix should be row-major, and in compressed format for
   * best performance. \sa SparseMatrix::makeCompressed().
   */
  template <typename InputMatrixType>
  void compute(const InputMatrixType &matrix) {
    if (m_symbolic) {
      NicsLU_Destroy(nicslu);
      m_isInitialized = false;
      nicslu = NULL;
      nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
      m_symbolic = 0;
      m_numeric = 0;
    }
    grab(matrix.derived());
    analyzePattern_impl();
    factorize_impl();
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems
   * having the same structure.
   *
   * \sa factorize(), compute()
   */
  template <typename InputMatrixType>
  void analyzePattern(const InputMatrixType &matrix) {
    if (m_symbolic) {
      NicsLU_Destroy(nicslu);
      // free(nicslu);
      m_isInitialized = false;
      nicslu = NULL;
      nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
      m_symbolic = 0;
      m_numeric = 0;
    }
    m_scale = 1;

    grab(matrix.derived());

    analyzePattern_impl();
  }

  /** Performs a symbolic decomposition on the sparcity of \a matrix.
   *
   * This function is particularly useful when solving for several problems
   * having the same structure.
   *
   * \sa factorize(), compute()
   */
  template <typename InputMatrixType, typename ListType>
  void analyzePatternPartial(const InputMatrixType &matrix, const ListType& variableList, const int mode) {
    if (m_symbolic) {
      NicsLU_Destroy(nicslu);
      m_isInitialized = false;
      nicslu = NULL;
      nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
      m_symbolic = 0;
      m_numeric = 0;
    }
    m_scale = 1;
    m_mode = mode;
    this->changedEntries = variableList;

    grab(matrix.derived());

    analyzePattern_impl();
  }

  /** Provides access to the control settings array used by NICSLU.
   *
   * See NICSLU documentation for details.
   */
  inline const SNicsLU *snicslu() const { return nicslu; }

  /** Provides access to the control settings array used by NICSLU.
   *
   * If this array contains NaN's, the default values are used.
   *
   * See NICSLU documentation for details.
   */
  inline SNicsLU *snicslu() { return nicslu; }

  /** Performs a numeric decomposition of \a matrix and computes factorization path
   *
   * The given matrix must has the same sparcity than the matrix on which the
   * pattern anylysis has been performed.
   *
   * \sa analyzePattern(), compute()
   */
  template <typename InputMatrixType>
  void factorize_partial(const InputMatrixType &matrix) {
    eigen_assert(m_analysisIsOk &&
                 "NICSLU: you must first call analyzePattern()");

    grab(matrix.derived());
    factorize_with_path_impl();
  }

  /** Performs a numeric decomposition of \a matrix and computes factorization path
   *
   * The given matrix must has the same sparcity than the matrix on which the
   * pattern anylysis has been performed.
   *
   * \sa analyzePattern(), compute()
   */
  template <typename InputMatrixType, typename ListType>
  void factorize_partial(const InputMatrixType &matrix, const ListType& list) {
    eigen_assert(m_analysisIsOk &&
                 "NICSLU: you must first call analyzePattern()");

    grab(matrix.derived());
    this->changedEntries = list;
    factorize_with_path_impl();
  }

  /** Performs a numeric decomposition of \a matrix
   *
   * The given matrix must has the same sparcity than the matrix on which the
   * pattern anylysis has been performed.
   *
   * \sa analyzePattern(), compute()
   */
  template <typename InputMatrixType>
  void factorize(const InputMatrixType &matrix) {
    eigen_assert(m_analysisIsOk &&
                 "NICSLU: you must first call analyzePattern()");

    grab(matrix.derived());
    factorize_impl();
  }

  template <typename InputMatrixType>
  void refactorize(const InputMatrixType &matrix) {
    eigen_assert(m_factorizationIsOk &&
                 "NICSLU: you must first call factorize()");
    grab(matrix.derived());
    refactorize_impl();
  }

  template <typename InputMatrixType>
  void refactorize_partial(const InputMatrixType &matrix) {
    eigen_assert(m_factorizationIsOk &&
                 "NICSLU: you must first call factorize()");
    grab(matrix.derived());
    refactorize_partial_impl();
  }

  /** \internal */
  template <typename BDerived, typename XDerived>
  bool _solve_impl(const MatrixBase<BDerived> &b,
                   MatrixBase<XDerived> &x) const;

#if 0 // not implemented yet
    Scalar determinant() const;

    void extractData() const;
#endif
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
protected:
  void init() {
    m_info = InvalidInput;
    m_isInitialized = false;
    m_numeric = 0;
    m_symbolic = 0;
    m_is_first_partial = 1;
    m_extractedDataAreDirty = true;
    nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
    NicsLU_Initialize(nicslu);
  }

  void analyzePattern_impl() {
    m_info = InvalidInput;
    m_analysisIsOk = false;
    m_factorizationIsOk = false;
    m_refactorizationIsOk = false;

    int nnz = mp_matrix.nonZeros();
    int okCreate, okAnalyze;
    if (!m_isInitialized)
      init();

    /* assemble matrix here */
    okCreate = NicsLU_CreateMatrix(
        nicslu, internal::convert_index<int>(mp_matrix.cols()), nnz,
        const_cast<Scalar *>(mp_matrix.valuePtr()),
        (unsigned int *)(mp_matrix.innerIndexPtr()),
        (unsigned int *)(mp_matrix.outerIndexPtr()));

    /* here: NICSLU parameters are set */
    /* scaling modes, etc. */
    nicslu->cfgi[0] = 1;
    nicslu->cfgi[1] = m_scale;
    nicslu->cfgi[10] = m_mode;

    /* setting pivoting tolerance for refatorization */
    nicslu->cfgf[31] = 1e-8;
    char *pivot_tolerance_env = getenv("NICSLU_PIVOT_TOLERANCE");
    if (pivot_tolerance_env != NULL) {
      double pivot_tolerance = atof(pivot_tolerance_env);
      if (pivot_tolerance > 0)
        nicslu->cfgf[31] = pivot_tolerance;
    }
    char *nicslu_do_mc64 = getenv("NICSLU_MC64");
    if (nicslu_do_mc64 != NULL) {
      nicslu->cfgi[1] = atoi(nicslu_do_mc64);
    }
    char *nicslu_scale = getenv("NICSLU_SCALE");
    if (nicslu_scale != NULL) {
      nicslu->cfgi[2] = atoi(nicslu_scale);
    }
    /* get both varying column and row entries, in case we want to do bra */
    uint__t* varyingr = NULL;
    uint__t* varyingc = NULL;
    uint__t numberOfVaryingEntries = this->changedEntries.size();
    if(numberOfVaryingEntries != 0)
    {
      varyingr = (uint__t*)calloc(numberOfVaryingEntries, sizeof(uint__t));
      varyingc = (uint__t*)calloc(numberOfVaryingEntries, sizeof(uint__t));
      uint__t counter = 0;
      for(std::pair<UInt, UInt> i : changedEntries){
        varyingr[counter] = i.first;
        varyingc[counter] = i.second;
        counter++;
      }
    }
    okAnalyze = NicsLU_Analyze(nicslu, varyingr, varyingc, numberOfVaryingEntries);

    if(varyingr)
      free(varyingr);
    if(varyingc)
      free(varyingc);

    if (okCreate == 0 && okAnalyze == 0) {
      m_symbolic = 1;
      m_isInitialized = true;
      m_info = Success;
      m_analysisIsOk = true;
      m_extractedDataAreDirty = true;
      m_is_first_partial = 1;
    }
  }

  void factorize_impl(){
    int numOk;
    Scalar* Az = const_cast<Scalar *>(mp_matrix.valuePtr());
    numOk = NicsLU_ResetMatrixValues(nicslu, Az);
    numOk = NicsLU_Factorize(nicslu);

    m_info = numOk == 0 ? Success : NumericalIssue;
    m_factorizationIsOk = numOk == 0 ? 1 : 0;
    m_numeric = numOk == 0 ? 1 : 0;
    m_extractedDataAreDirty = true;
    m_is_first_partial = 1;
  }

  void factorize_with_path_impl() {
    int numOk;
    Scalar* Az = const_cast<Scalar *>(mp_matrix.valuePtr());
    numOk = NicsLU_ResetMatrixValues(nicslu, Az);
    numOk = NicsLU_Factorize(nicslu);

    /* Explanation for modes: nicslu->cfgi[10] = mode
     * mode 0: partial refact. with amd ordering (factorization path)
     * mode 1: partial refact. with specialized ordering (factorization path - MD-NV)
     * mode 2: partial refact. with bottom-right-arranging (restarting)
     * mode 3: restarting partial refact. (aka "canadian method") (restarting)
     */

    if(nicslu->cfgi[10] == 0 || nicslu->cfgi[10] == 1)
    {
      /* factorization path required here */
      NicsLU_compute_path(nicslu);
    }
    else
    {
      /* no factorization path required, doing refactorization restart */
      /* this function essentially tells NICSLU where to restart the refactorization */
      NicsLU_RefactorizationRestart(nicslu);
    }

    m_info = numOk == 0 ? Success : NumericalIssue;
    m_factorizationIsOk = numOk == 0 ? 1 : 0;
    m_numeric = numOk == 0 ? 1 : 0;
    m_extractedDataAreDirty = true;
    m_is_first_partial = 1;
  }

  void refactorize_partial_impl(){
    // Make sure that NicsLU_Factorize was called before this!
    eigen_assert(
        m_factorizationIsOk &&
        "The decomposition is not in a valid state for refactorization, you "
        "must first call either compute() or analyzePattern()/factorize()");
    int numOk;
#ifdef EIGEN_DUMP
	static int counter = 1;
	if(counter == 1 || counter == m_limit)
	{
		printMTX(mp_matrix, counter);	
	}
	counter++;
#endif

    if(nicslu->cfgi[10] == 0 || nicslu->cfgi[10] == 1)
    {
      /* factorization path mode */

      /* check if something went terribly wrong... 
       * this is necessary if the user factorises a matrix with 
       * a different nnz pattern without calling the destructror 
       * of this class. Then, the old orderings and factorizations
       * are still in the memory. */
      if (mp_matrix.nonZeros() != nicslu->nnz) {
        /* re-do everything... */
        analyzePattern_impl();
        numOk = NicsLU_Factorize(nicslu);
        NicsLU_compute_path(nicslu);
        m_factorizationIsOk = numOk == 0 ? 1 : 0;
      } else { 
        /* get new matrix values */
        Scalar* Ax = const_cast<Scalar*>(mp_matrix.valuePtr());

        /* Refactorize partially with new values */
        numOk = NicsLU_PartialReFactorize(nicslu, Ax);

        /* check whether a pivot became too large or too small */
        if (numOk == NICSLU_NUMERIC_OVERFLOW) {
          /* if so, reset matrix values and re-do factorization
           * only needs to Reset Matrix Values, if analyze_pattern is not called
           */
          numOk = NicsLU_ResetMatrixValues(nicslu, Ax);
          numOk = NicsLU_Factorize(nicslu);
          m_factorizationIsOk = numOk == 0 ? 1 : 0;
        }
      }
    }
    else
    {
      /* 
        restarting partial refactorisation
        either BRA or AMD ordering ("canadian method")
      */

      // check if something went terribly wrong...
      if (mp_matrix.nonZeros() != nicslu->nnz) {
        /*  */
        analyzePattern_impl();
        numOk = NicsLU_Factorize(nicslu);
        NicsLU_RefactorizationRestart(nicslu);
        m_factorizationIsOk = numOk == 0 ? 1 : 0;
      } else {
        // get new matrix values
        Scalar* Ax = const_cast<Scalar*>(mp_matrix.valuePtr());

        // Refactorize with new values
        numOk = NicsLU_FPartialReFactorize(nicslu, Ax);

        // check whether a pivot became too large or too small
        if (numOk == NICSLU_NUMERIC_OVERFLOW) {
          // if so, reset matrix values and re-do computation
          // only needs to Reset Matrix Values, if analyze_pattern is not called
          numOk = NicsLU_ResetMatrixValues(nicslu, Ax);
          numOk = NicsLU_Factorize(nicslu);
          m_factorizationIsOk = numOk == 0 ? 1 : 0;
        }
      }
    }
    m_info = numOk == 0 ? Success : NumericalIssue;
    m_refactorizationIsOk = numOk == 0 ? 1 : 0;
    m_numeric = numOk == 0 ? 1 : 0;
    m_extractedDataAreDirty = true;
    eigen_assert(
        m_factorizationIsOk &&
        "The decomposition is not in a valid state for refactorization");
  }

  void refactorize_impl() {
    // Make sure that NicsLU_Factorize was called before this!
    eigen_assert(
        m_factorizationIsOk &&
        "The decomposition is not in a valid state for refactorization, you "
        "must first call either compute() or analyzePattern()/factorize()");
    int numOk;

    // check if something went terribly wrong...
    if (mp_matrix.nonZeros() != nicslu->nnz) {
      analyzePattern_impl();
      numOk = NicsLU_Factorize(nicslu);
      m_is_first_partial = 1;
      m_factorizationIsOk = numOk == 0 ? 1 : 0;
    } else {
      // get new matrix values
      Scalar* Ax = const_cast<Scalar*>(mp_matrix.valuePtr());

      // Refactorize with new values
      numOk = NicsLU_ReFactorize(nicslu, Ax);

      // check whether a pivot became too large or too small
      if (numOk == NICSLU_NUMERIC_OVERFLOW) {
        // if so, reset matrix values and re-do computation
        // only needs to Reset Matrix Values, if analyze_pattern is not called
        numOk = NicsLU_ResetMatrixValues(nicslu, Ax);
        numOk = NicsLU_Factorize(nicslu);
        m_factorizationIsOk = numOk == 0 ? 1 : 0;
      }
    }
    m_info = numOk == 0 ? Success : NumericalIssue;
    m_refactorizationIsOk = numOk == 0 ? 1 : 0;
    m_numeric = numOk == 0 ? 1 : 0;
    m_extractedDataAreDirty = true;
  }

  template <typename MatrixDerived>
  void grab(const EigenBase<MatrixDerived> &A) {
    mp_matrix.~NICSLUMatrixRef();
    ::new (&mp_matrix) NICSLUMatrixRef(A.derived());
  }

  void grab(const NICSLUMatrixRef &A) {
    if (&(A.derived()) != &mp_matrix) {
      mp_matrix.~NICSLUMatrixRef();
      ::new (&mp_matrix) NICSLUMatrixRef(A);
    }
  }

  // cached data to reduce reallocation, etc.
#if 0 // not implemented yet
    mutable LUMatrixType m_l;
    mutable LUMatrixType m_u;
    mutable IntColVectorType m_p;
    mutable IntRowVectorType m_q;
#endif

  NICSLUMatrixType m_dummy;
  NICSLUMatrixRef mp_matrix;

  SNicsLU *nicslu;
  mutable ComputationInfo m_info;
  int m_factorizationIsOk;
  int m_refactorizationIsOk;
  int m_analysisIsOk;
  mutable bool m_extractedDataAreDirty;
  int m_symbolic;
  int m_numeric;
  int m_is_first_partial;
  // do scaling?
  int m_scale;
  /* which permutation mode?
  * 0: normal AMD
  * 1: factorization path shortening
  * 2: bottom-right arranging
  * 3: partial refactorisation by refactorisation restart
  */
  int m_mode;
  int m_limit;

private:
  //    NICSLU(const NICSLU& ) { }
};

#if 0 // not implemented yet
template<typename MatrixType>
void NICSLU<MatrixType>::extractData() const
{
  if (m_extractedDataAreDirty)
  {
     eigen_assert(false && "NICSLU: extractData Not Yet Implemented");
  }
}

template<typename MatrixType>
typename NICSLU<MatrixType>::Scalar NICSLU<MatrixType>::determinant() const
{
  eigen_assert(false && "NICSLU: extractData Not Yet Implemented");
  return Scalar();
}
#endif

template <typename MatrixType>
template <typename BDerived, typename XDerived>
bool NICSLU<MatrixType>::_solve_impl(const MatrixBase<BDerived> &b,
                                     MatrixBase<XDerived> &x) const {
  // Index rhsCols = b.cols();
  EIGEN_STATIC_ASSERT((XDerived::Flags & RowMajorBit) == 0,
                      THIS_METHOD_IS_ONLY_FOR_ROW_MAJOR_MATRICES);
  eigen_assert(
      m_factorizationIsOk &&
      "The decomposition is not in a valid state for solving, you must first "
      "call either compute() or analyzePattern()/factorize()");

  x = b;
  // MatrixBase<XDerived> bcopy = x;
  int info = NicsLU_Solve(nicslu, x.const_cast_derived().data());
  // info = NicsLU_Refine(nicslu, x.const_cast_derived().data(),
  // b.const_cast_derived().data(), nicslu->cfgf[0], 3);
  m_info = info == 0 ? Success : NumericalIssue;
  return m_info == Success;
}

} // end namespace Eigen

#endif // EIGEN_NICSLUSUPPORT_H
