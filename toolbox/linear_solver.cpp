#include "linear_solver.h"


bool LDLTSolver::setSystem(const Eigen::SparseMatrix<double>& A)
{
    if(!initialized)
    {
        chol.analyzePattern(A);
        if(chol.info()) return chol.info();
        initialized = true;
    }
    
    chol.factorize(A);
    
    return chol.info() == 0 && chol.vectorD().minCoeff() > -1e-10;
}

bool LLTSolver::setSystem(const Eigen::SparseMatrix<double>& A)
{
    if(!initialized)
    {
        chol.analyzePattern(A);
        if(chol.info()) return chol.info();
        initialized = true;
    }
    
    chol.compute(A);
    return chol.info() != Eigen::NumericalIssue;
}

int CGSolver::setSystem(const Eigen::SparseMatrix<double>& A)
{
    cg.compute(A);
    return cg.info();
}


int LUSolver::setSystem(const Eigen::SparseMatrix<double>& A)
{
    lu.compute(A);
    return lu.info();
}


#ifdef HAVE_CHOLMOD

int CholmodSolver::setSystem(const Eigen::SparseMatrix<double>& A)
{
    if(!initialized)
    {
        chol.analyzePattern(A);
        if(chol.info()) return chol.info();
        initialized = true;
    }
    
    chol.factorize(A);
    return chol.info() == 0;
}


#endif
