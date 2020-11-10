#ifndef linear_solver_hpp
#define linear_solver_hpp

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

#ifdef HAVE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif


class LDLTSolver 
{
    bool initialized = false;
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;
    
public:
    LDLTSolver() {};
       
    bool setSystem(const Eigen::SparseMatrix<double>& A);

    int solve(const Eigen::VectorXd& b, Eigen::VectorXd& x)
    {
        x = chol.solve(b);
        return chol.info();
    }

    int solve(const Eigen::MatrixXd& b,  Eigen::MatrixXd& x )
    {
        x = chol.solve(b);
        return chol.info();
    }
};

class LLTSolver
{
    bool initialized = false;
    
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> chol;
    
public:
    LLTSolver() {};
       
    bool setSystem(const Eigen::SparseMatrix<double>& A);
    
    template<typename rhs>
    int solve(const Eigen::MatrixBase<rhs>& b, Eigen::MatrixBase<rhs>& x)
    {
        x = chol.solve(b);
        return chol.info();
    }
};


class CGSolver
{
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    
public:
    CGSolver() {};
    
    int setSystem(const Eigen::SparseMatrix<double>& A);
    
    template<typename rhs>
    int solve(const Eigen::MatrixBase<rhs>& b, Eigen::MatrixBase<rhs>& x)
    {
        x = cg.solve(b);
        return cg.info();
    }
};


class LUSolver
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;

public:
    LUSolver() {};

    int setSystem(const Eigen::SparseMatrix<double>& A);

    template<typename rhs>
    int solve(const Eigen::MatrixBase<rhs>& b, Eigen::MatrixBase<rhs>& x)
    {
        x = lu.solve(b);
        return lu.info();
    }
};
 

#ifdef HAVE_CHOLMOD

class CholmodSolver 
{
    bool initialized = false;
    
   // Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> chol;
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> chol;
    
public:
    CholmodSolver() {};
    
    int setSystem(const Eigen::SparseMatrix<double>& A);
    
    
    int solve(const Eigen::VectorXd& b, Eigen::VectorXd& x)
    {
        x = chol.solve(b);
        return chol.info();
    }
    
    int solve(const Eigen::MatrixXd& b,  Eigen::MatrixXd& x)
    {
        x = chol.solve(b);
        return chol.info();
    }
};

#endif

#endif
