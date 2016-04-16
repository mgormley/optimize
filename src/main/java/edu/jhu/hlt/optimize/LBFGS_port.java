/*
 *      Java library of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * Copyright (c) 2016 Matt Gormley
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package edu.jhu.hlt.optimize;

import static edu.jhu.hlt.optimize.LBFGS_port.LineSearchAlg.LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
import static edu.jhu.hlt.optimize.LBFGS_port.LineSearchAlg.LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
import static edu.jhu.hlt.optimize.LBFGS_port.LineSearchAlg.*;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INCORRECT_TMINMAX;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INCREASEGRADIENT;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALIDPARAMETERS;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_DELTA;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_EPSILON;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_FTOL;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_GTOL;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_LINESEARCH;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.*;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_MAXSTEP;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_MINSTEP;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_N;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_ORTHANTWISE;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_ORTHANTWISE_END;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_ORTHANTWISE_START;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_TESTPERIOD;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_WOLFE;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_INVALID_XTOL;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_MAXIMUMITERATION;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_MAXIMUMLINESEARCH;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_MAXIMUMSTEP;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_MINIMUMSTEP;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_OUTOFINTERVAL;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_ROUNDING_ERROR;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGSERR_WIDTHTOOSMALL;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGS_ALREADY_MINIMIZED;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGS_CONTINUE;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGS_STOP;
import static edu.jhu.hlt.optimize.LBFGS_port.StatusCode.LBFGS_SUCCESS;

import edu.jhu.prim.Primitives.MutableDouble;
import edu.jhu.prim.Primitives.MutableInt;

/**
 * This is a port of libLBFGS from C to Java.
 * The original C source code is available at:
 * https://github.com/chokkan/liblbfgs
 * 
 * libLBFGS is a C port of the implementation of Limited-memory
 * Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal. 
 * The original FORTRAN source code is available at: 
 * http://www.ece.northwestern.edu/~nocedal/lbfgs.html
 * 
 * @author mgormley
 */
public class LBFGS_port {
    
    /** 
     * \addtogroup liblbfgs_api libLBFGS API
     * @{
     *
     *  The libLBFGS API.
     */

    /**
     * Return values of lbfgs().
     * 
     *  Roughly speaking, a negative value indicates an error.
     */
    public enum StatusCode {
        /** L-BFGS reaches convergence. */
        LBFGS_SUCCESS(0),
        LBFGS_CONVERGENCE(0),
        LBFGS_STOP(1),
        /** The initial variables already minimize the objective function. */
        LBFGS_ALREADY_MINIMIZED(2),
        /** For internal use only: optimization should continue. */
        LBFGS_CONTINUE(0),
        /** For internal use only: line search success. */
        LBFGS_LS_SUCCESS(0),
    
        /** Unknown error. */
        LBFGSERR_UNKNOWNERROR(-1024),
        /** Logic error. */
        LBFGSERR_LOGICERROR(-1023),
        /** Insufficient memory. */
        LBFGSERR_OUTOFMEMORY(-1022),
        /** The minimization process has been canceled. */
        LBFGSERR_CANCELED(-1021),
        /** Invalid number of variables specified. */
        LBFGSERR_INVALID_N(-1020),
        /** Invalid number of variables (for SSE) specified. */
        LBFGSERR_INVALID_N_SSE(-1019),
        /** The array x must be aligned to 16 (for SSE). */
        LBFGSERR_INVALID_X_SSE(-1018),
        /** Invalid parameter lbfgs_parameter_t::epsilon specified. */
        LBFGSERR_INVALID_EPSILON(-1017),
        /** Invalid parameter lbfgs_parameter_t::past specified. */
        LBFGSERR_INVALID_TESTPERIOD(-1016),
        /** Invalid parameter lbfgs_parameter_t::delta specified. */
        LBFGSERR_INVALID_DELTA(-1015),
        /** Invalid parameter lbfgs_parameter_t::linesearch specified. */
        LBFGSERR_INVALID_LINESEARCH(-1014),
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        LBFGSERR_INVALID_MINSTEP(-1013),
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        LBFGSERR_INVALID_MAXSTEP(-1012),
        /** Invalid parameter lbfgs_parameter_t::ftol specified. */
        LBFGSERR_INVALID_FTOL(-1011),
        /** Invalid parameter lbfgs_parameter_t::wolfe specified. */
        LBFGSERR_INVALID_WOLFE(-1010),
        /** Invalid parameter lbfgs_parameter_t::gtol specified. */
        LBFGSERR_INVALID_GTOL(-1009),
        /** Invalid parameter lbfgs_parameter_t::xtol specified. */
        LBFGSERR_INVALID_XTOL(-1008),
        /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
        LBFGSERR_INVALID_MAXLINESEARCH(-1007),
        /** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
        LBFGSERR_INVALID_ORTHANTWISE(-1006),
        /** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
        LBFGSERR_INVALID_ORTHANTWISE_START(-1005),
        /** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
        LBFGSERR_INVALID_ORTHANTWISE_END(-1004),
        /** The line-search step went out of the interval of uncertainty. */
        LBFGSERR_OUTOFINTERVAL(-1003),
        /** A logic error occurred; alternatively, the interval of uncertainty
            became too small. */
        LBFGSERR_INCORRECT_TMINMAX(-1002),
        /** A rounding error occurred; alternatively, no line-search step
            satisfies the sufficient decrease and curvature conditions. */
        LBFGSERR_ROUNDING_ERROR(-1001),
        /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
        LBFGSERR_MINIMUMSTEP(-1000),
        /** The line-search step became larger than lbfgs_parameter_t::max_step. */
        LBFGSERR_MAXIMUMSTEP(-999),
        /** The line-search routine reaches the maximum number of evaluations. */
        LBFGSERR_MAXIMUMLINESEARCH(-998),
        /** The algorithm routine reaches the maximum number of iterations. */
        LBFGSERR_MAXIMUMITERATION(-997),
        /** Relative width of the interval of uncertainty is at most
            lbfgs_parameter_t::xtol. */
        LBFGSERR_WIDTHTOOSMALL(-996),
        /** A logic error (negative line-search step) occurred. */
        LBFGSERR_INVALIDPARAMETERS(-995),
        /** The current search direction increases the objective function value. */
        LBFGSERR_INCREASEGRADIENT(-994);
        
        public final int ret;
        private StatusCode(int ret) { this.ret = ret; }
    }

    /**
     * Line search algorithms.
     */
    public enum LineSearchAlg {
        
        /** MoreThuente method proposd by More and Thuente. */
        LBFGS_LINESEARCH_MORETHUENTE(0),
        
        /**
         * Backtracking method with the Armijo condition.
         *  The backtracking method finds the step length such that it satisfies
         *  the sufficient decrease (Armijo) condition,
         *    - f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d,
         *
         *  where x is the current point, d is the current search direction, and
         *  a is the step length.
         */
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO(1),
        
        /**
         * Backtracking method with regular Wolfe condition.
         *  The backtracking method finds the step length such that it satisfies
         *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
         *  and the curvature condition,
         *    - g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
         *
         *  where x is the current point, d is the current search direction, and
         *  a is the step length.
         */
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE(2),
        
        /**
         * Backtracking method with strong Wolfe condition.
         *  The backtracking method finds the step length such that it satisfies
         *  both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
         *  and the following condition,
         *    - |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
         *
         *  where x is the current point, d is the current search direction, and
         *  a is the step length.
         */
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE(3);

        public final int v;
        private LineSearchAlg(int v) { this.v = v; }
        
    }
    
    /**
     * L-BFGS optimization parameters.
     *  Call lbfgs_parameter_init() function to initialize parameters to the
     *  default values.
     */
    public static class lbfgs_parameter_t {
                
        public lbfgs_parameter_t() { }
        
        public lbfgs_parameter_t(lbfgs_parameter_t other) {
            super();
            this.m = other.m;
            this.epsilon = other.epsilon;
            this.past = other.past;
            this.delta = other.delta;
            this.max_iterations = other.max_iterations;
            this.linesearch = other.linesearch;
            this.max_linesearch = other.max_linesearch;
            this.min_step = other.min_step;
            this.max_step = other.max_step;
            this.ftol = other.ftol;
            this.wolfe = other.wolfe;
            this.gtol = other.gtol;
            this.xtol = other.xtol;
            this.orthantwise_c = other.orthantwise_c;
            this.orthantwise_start = other.orthantwise_start;
            this.orthantwise_end = other.orthantwise_end;
        }

        /**
         * The number of corrections to approximate the inverse hessian matrix.
         *  The L-BFGS routine stores the computation results of previous \ref m
         *  iterations to approximate the inverse hessian matrix of the current
         *  iteration. This parameter controls the size of the limited memories
         *  (corrections). The default value is \c 6. Values less than \c 3 are
         *  not recommended. Large values will result in excessive computing time.
         */
        int             m = 6;

        /**
         * Epsilon for convergence test.
         *  This parameter determines the accuracy with which the solution is to
         *  be found. A minimization terminates when
         *      ||g|| < \ref epsilon * max(1, ||x||),
         *  where ||.|| denotes the Euclidean (L2) norm. The default value is
         *  \c 1e-5.
         */
        double epsilon = 1e-5;

        /**
         * Distance for delta-based convergence test.
         *  This parameter determines the distance, in iterations, to compute
         *  the rate of decrease of the objective function. If the value of this
         *  parameter is zero, the library does not perform the delta-based
         *  convergence test. The default value is \c 0.
         */
        int             past = 0;

        /**
         * Delta for convergence test.
         *  This parameter determines the minimum rate of decrease of the
         *  objective function. The library stops iterations when the
         *  following condition is met:
         *      (f' - f) / f < \ref delta,
         *  where f' is the objective value of \ref past iterations ago, and f is
         *  the objective value of the current iteration.
         *  The default value is \c 0.
         */
        double delta = 0;

        /**
         * The maximum number of iterations.
         *  The lbfgs() function terminates an optimization process with
         *  ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
         *  exceedes this parameter. Setting this parameter to zero continues an
         *  optimization process until a convergence or error. The default value
         *  is \c 0.
         */
        int             max_iterations = 0;

        /**
         * The line search algorithm.
         *  This parameter specifies a line search algorithm to be used by the
         *  L-BFGS routine.
         */
        LineSearchAlg linesearch = LBFGS_LINESEARCH_MORETHUENTE;

        /**
         * The maximum number of trials for the line search.
         *  This parameter controls the number of function and gradients evaluations
         *  per iteration for the line search routine. The default value is \c 20.
         */
        int             max_linesearch = 20; // was 40

        /**
         * The minimum step of the line search routine.
         *  The default value is \c 1e-20. This value need not be modified unless
         *  the exponents are too large for the machine being used, or unless the
         *  problem is extremely badly scaled (in which case the exponents should
         *  be increased).
         */
        double min_step = 1e-20;

        /**
         * The maximum step of the line search.
         *  The default value is \c 1e+20. This value need not be modified unless
         *  the exponents are too large for the machine being used, or unless the
         *  problem is extremely badly scaled (in which case the exponents should
         *  be increased).
         */
        double max_step = 1e20;

        /**
         * A parameter to control the accuracy of the line search routine.
         *  The default value is \c 1e-4. This parameter should be greater
         *  than zero and smaller than \c 0.5.
         */
        double ftol = 1e-4;

        /**
         * A coefficient for the Wolfe condition.
         *  This parameter is valid only when the backtracking line-search
         *  algorithm is used with the Wolfe condition,
         *  ::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
         *  ::LBFGS_LINESEARCH_BACKTRACKING_WOLFE .
         *  The default value is \c 0.9. This parameter should be greater
         *  the \ref ftol parameter and smaller than \c 1.0.
         */
        double wolfe = 0.9;

        /**
         * A parameter to control the accuracy of the line search routine.
         *  The default value is \c 0.9. If the function and gradient
         *  evaluations are inexpensive with respect to the cost of the
         *  iteration (which is sometimes the case when solving very large
         *  problems) it may be advantageous to set this parameter to a small
         *  value. A typical small value is \c 0.1. This parameter shuold be
         *  greater than the \ref ftol parameter (\c 1e-4) and smaller than
         *  \c 1.0.
         */
        double gtol = 0.9;

        /**
         * The machine precision for floating-point values.
         *  This parameter must be a positive value set by a client program to
         *  estimate the machine precision. The line search routine will terminate
         *  with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
         *  of the interval of uncertainty is less than this parameter.
         */
        double xtol = 1e-16;

        /**
         * Coeefficient for the L1 norm of variables.
         *  This parameter should be set to zero for standard minimization
         *  problems. Setting this parameter to a positive value activates
         *  Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
         *  minimizes the objective function F(x) combined with the L1 norm |x|
         *  of the variables, {F(x) + C |x|}. This parameter is the coeefficient
         *  for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
         *  zero, the library modifies function and gradient evaluations from
         *  a client program suitably; a client program thus have only to return
         *  the function value F(x) and gradients G(x) as usual. The default value
         *  is zero.
         */
        double orthantwise_c = 0.0;

        /**
         * Start index for computing L1 norm of the variables.
         *  This parameter is valid only for OWL-QN method
         *  (i.e., \ref orthantwise_c != 0). This parameter b (0 <= b < N)
         *  specifies the index number from which the library computes the
         *  L1 norm of the variables x,
         *      |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
         *  In other words, variables x_1, ..., x_{b-1} are not used for
         *  computing the L1 norm. Setting b (0 < b < N), one can protect
         *  variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
         *  regression) from being regularized. The default value is zero.
         */
        int             orthantwise_start = 0;

        /**
         * End index for computing L1 norm of the variables.
         *  This parameter is valid only for OWL-QN method
         *  (i.e., \ref orthantwise_c != 0). This parameter e (0 < e <= N)
         *  specifies the index number at which the library stops computing the
         *  L1 norm of the variables x,
         */
        int             orthantwise_end = -1;
    }


    public static abstract class callback_data_t {
        int n; // TODO: Remove this.
        Object instance;
        
        /**
         * Callback interface to provide objective function and gradient evaluations.
         *
         *  The lbfgs() function call this function to obtain the values of objective
         *  function and its gradients when needed. A client program must implement
         *  this function to evaluate the values of the objective function and its
         *  gradients, given current values of variables.
         *  
         *  @param  instance    The user data sent for lbfgs() function by the client.
         *  @param  x           The current values of variables.
         *  @param  g           The gradient vector. The callback function must compute
         *                      the gradient values for the current variables.
         *  @param  n           The number of variables.
         *  @param  step        The current step of the line search routine.
         *  @retval double The value of the objective function for the current
         *                          variables.
         */
        abstract double proc_evaluate(
            Object instance,
            final double[] x,
            double[] g,
            final int n,
            final double step
            );
    
        /**
         * Callback interface to receive the progress of the optimization process.
         *
         *  The lbfgs() function call this function for each iteration. Implementing
         *  this function, a client program can store or display the current progress
         *  of the optimization process.
         *
         *  @param  instance    The user data sent for lbfgs() function by the client.
         *  @param  x           The current values of variables.
         *  @param  g           The current gradient values of variables.
         *  @param  fx          The current value of the objective function.
         *  @param  xnorm       The Euclidean norm of the variables.
         *  @param  gnorm       The Euclidean norm of the gradients.
         *  @param  step        The line-search step used for this iteration.
         *  @param  n           The number of variables.
         *  @param  k           The iteration count.
         *  @param  ls          The number of evaluations called for this iteration.
         *  @retval int         Zero to continue the optimization process. Returning a
         *                      non-zero value will cancel the optimization process.
         */
        abstract StatusCode proc_progress(
            Object instance,
            final double[] x,
            final double[] g,
            final double fx,
            final double xnorm,
            final double gnorm,
            final double step,
            int n,
            int k,
            int ls
            );
        

        /*
        A user must implement a function compatible with ::lbfgs_evaluate_t (evaluation
        callback) and pass the pointer to the callback function to lbfgs() arguments.
        Similarly, a user can implement a function compatible with ::lbfgs_progress_t
        (progress callback) to obtain the current progress (e.g., variables, function
        value, ||G||, etc) and to cancel the iteration process if necessary.
        Implementation of a progress callback is optional: a user can pass \c null if
        progress notification is not necessary.
    
        In addition, a user must preserve two requirements:
            - The number of variables must be multiples of 16 (this is not 4).
            - The memory block of variable array ::x must be aligned to 16.
    
        This algorithm terminates an optimization
        when:
    
            ||G|| < \epsilon \cdot \max(1, ||x||) .
    
        In this formula, ||.|| denotes the Euclidean norm.
        */
    
    }

    /** @} */



    /**
    @mainpage libLBFGS: a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)

    @section intro Introduction

    This library is a C port of the implementation of Limited-memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
    The original FORTRAN source code is available at:
    http://www.ece.northwestern.edu/~nocedal/lbfgs.html

    The L-BFGS method solves the unconstrainted minimization problem,

    <pre>
        minimize F(x), x = (x1, x2, ..., xN),
    </pre>

    only if the objective function F(x) and its gradient G(x) are computable. The
    well-known Newton's method requires computation of the inverse of the hessian
    matrix of the objective function. However, the computational cost for the
    inverse hessian matrix is expensive especially when the objective function
    takes a large number of variables. The L-BFGS method iteratively finds a
    minimizer by approximating the inverse hessian matrix by information from last
    m iterations. This innovation saves the memory storage and computational time
    drastically for large-scaled problems.

    Among the various ports of L-BFGS, this library provides several features:
    - <b>Optimization with L1-norm (Orthant-Wise Limited-memory Quasi-Newton
      (OWL-QN) method)</b>:
      In addition to standard minimization problems, the library can minimize
      a function F(x) combined with L1-norm |x| of the variables,
      {F(x) + C |x|}, where C is a constant scalar parameter. This feature is
      useful for estimating parameters of sparse log-linear models (e.g.,
      logistic regression and maximum entropy) with L1-regularization (or
      Laplacian prior).
    - <b>Clean C code</b>:
      Unlike C codes generated automatically by f2c (Fortran 77 into C converter),
      this port includes changes based on my interpretations, improvements,
      optimizations, and clean-ups so that the ported code would be well-suited
      for a C code. In addition to comments inherited from the original code,
      a number of comments were added through my interpretations.
    - <b>Callback interface</b>:
      The library receives function and gradient values via a callback interface.
      The library also notifies the progress of the optimization by invoking a
      callback function. In the original implementation, a user had to set
      function and gradient values every time the function returns for obtaining
      updated values.
    - <b>Thread safe</b>:
      The library is thread-safe, which is the secondary gain from the callback
      interface.
    - <b>Cross platform.</b> The source code can be compiled on Microsoft Visual
      Studio 2010, GNU C Compiler (gcc), etc.
    - <b>Configurable precision</b>: A user can choose single-precision (float)
      or double-precision (double) accuracy by changing ::LBFGS_FLOAT macro.
    - <b>SSE/SSE2 optimization</b>:
      This library includes SSE/SSE2 optimization (written in compiler intrinsics)
      for vector arithmetic operations on Intel/AMD processors. The library uses
      SSE for float values and SSE2 for double values. The SSE/SSE2 optimization
      routine is disabled by default.

    This library is used by:
    - <a href="http://www.chokkan.org/software/crfsuite/">CRFsuite: A fast implementation of Conditional Random Fields (CRFs)</a>
    - <a href="http://www.chokkan.org/software/classias/">Classias: A collection of machine-learning algorithms for classification</a>
    - <a href="http://www.public.iastate.edu/~gdancik/mlegp/">mlegp: an R package for maximum likelihood estimates for Gaussian processes</a>
    - <a href="http://infmath.uibk.ac.at/~matthiasf/imaging2/">imaging2: the imaging2 class library</a>

    @section download Download

    - <a href="https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz">Source code</a>
    - <a href="https://github.com/chokkan/liblbfgs">GitHub repository</a>

    libLBFGS is distributed under the term of the
    <a href="http://opensource.org/licenses/mit-license.php">MIT license</a>.

    @section modules Third-party modules
    - <a href="http://cran.r-project.org/web/packages/lbfgs/index.html">lbfgs: Limited-memory BFGS Optimization (a wrapper for R)</a> maintained by Antonio Coppola.
    - <a href="http://search.cpan.org/~laye/Algorithm-LBFGS-0.16/">Algorithm::LBFGS - Perl extension for L-BFGS</a> maintained by Lei Sun.
    - <a href="http://www.cs.kuleuven.be/~bernd/yap-lbfgs/">YAP-LBFGS (an interface to call libLBFGS from YAP Prolog)</a> maintained by Bernd Gutmann.

    @section changelog History
    - Version 1.10 (2010-12-22):
        - Fixed compiling errors on Mac OS X; this patch was kindly submitted by
          Nic Schraudolph.
        - Reduced compiling warnings on Mac OS X; this patch was kindly submitted
          by Tamas Nepusz.
        - Replaced memalign() with posix_memalign().
        - Updated solution and project files for Microsoft Visual Studio 2010.
    - Version 1.9 (2010-01-29):
        - Fixed a mistake in checking the validity of the parameters "ftol" and
          "wolfe"; this was discovered by Kevin S. Van Horn.
    - Version 1.8 (2009-07-13):
        - Accepted the patch submitted by Takashi Imamichi;
          the backtracking method now has three criteria for choosing the step
          length:
            - ::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO: sufficient decrease (Armijo)
              condition only
            - ::LBFGS_LINESEARCH_BACKTRACKING_WOLFE: regular Wolfe condition
              (sufficient decrease condition + curvature condition)
            - ::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE: strong Wolfe condition
        - Updated the documentation to explain the above three criteria.
    - Version 1.7 (2009-02-28):
        - Improved OWL-QN routines for stability.
        - Removed the support of OWL-QN method in MoreThuente algorithm because
          it accidentally fails in early stages of iterations for some objectives.
          Because of this change, <b>the OW-LQN method must be used with the
          backtracking algorithm (::LBFGS_LINESEARCH_BACKTRACKING)</b>, or the
          library returns ::LBFGSERR_INVALID_LINESEARCH.
        - Renamed line search algorithms as follows:
            - ::LBFGS_LINESEARCH_BACKTRACKING: regular Wolfe condition.
            - ::LBFGS_LINESEARCH_BACKTRACKING_LOOSE: regular Wolfe condition.
            - ::LBFGS_LINESEARCH_BACKTRACKING_STRONG: strong Wolfe condition.
        - Source code clean-up.
    - Version 1.6 (2008-11-02):
        - Improved line-search algorithm with strong Wolfe condition, which was
          contributed by Takashi Imamichi. This routine is now default for
          ::LBFGS_LINESEARCH_BACKTRACKING. The previous line search algorithm
          with regular Wolfe condition is still available as
          ::LBFGS_LINESEARCH_BACKTRACKING_LOOSE.
        - Configurable stop index for L1-norm computation. A member variable
          ::lbfgs_parameter_t::orthantwise_end was added to specify the index
          number at which the library stops computing the L1 norm of the
          variables. This is useful to prevent some variables from being
          regularized by the OW-LQN method.
        - A sample program written in C++ (sample/sample.cpp).
    - Version 1.5 (2008-07-10):
        - Configurable starting index for L1-norm computation. A member variable
          ::lbfgs_parameter_t::orthantwise_start was added to specify the index
          number from which the library computes the L1 norm of the variables.
          This is useful to prevent some variables from being regularized by the
          OWL-QN method.
        - Fixed a zero-division error when the initial variables have already
          been a minimizer (reported by Takashi Imamichi). In this case, the
          library returns ::LBFGS_ALREADY_MINIMIZED status code.
        - Defined ::LBFGS_SUCCESS status code as zero; removed unused constants,
          LBFGSFALSE and LBFGSTRUE.
        - Fixed a compile error in an implicit down-cast.
    - Version 1.4 (2008-04-25):
        - Configurable line search algorithms. A member variable
          ::lbfgs_parameter_t::linesearch was added to choose either MoreThuente
          method (::LBFGS_LINESEARCH_MORETHUENTE) or backtracking algorithm
          (::LBFGS_LINESEARCH_BACKTRACKING).
        - Fixed a bug: the previous version did not compute psuedo-gradients
          properly in the line search routines for OWL-QN. This bug might quit
          an iteration process too early when the OWL-QN routine was activated
          (0 < ::lbfgs_parameter_t::orthantwise_c).
        - Configure script for POSIX environments.
        - SSE/SSE2 optimizations with GCC.
        - New functions ::lbfgs_malloc and ::lbfgs_free to use SSE/SSE2 routines
          transparently. It is uncessary to use these functions for libLBFGS built
          without SSE/SSE2 routines; you can still use any memory allocators if
          SSE/SSE2 routines are disabled in libLBFGS.
    - Version 1.3 (2007-12-16):
        - An API change. An argument was added to lbfgs() function to receive the
          final value of the objective function. This argument can be set to
          \c null if the final value is unnecessary.
        - Fixed a null-pointer bug in the sample code (reported by Takashi Imamichi).
        - Added build scripts for Microsoft Visual Studio 2005 and GCC.
        - Added README file.
    - Version 1.2 (2007-12-13):
        - Fixed a serious bug in orthant-wise L-BFGS.
          An important variable was used without initialization.
    - Version 1.1 (2007-12-01):
        - Implemented orthant-wise L-BFGS.
        - Implemented lbfgs_parameter_init() function.
        - Fixed several bugs.
        - API documentation.
    - Version 1.0 (2007-09-20):
        - Initial release.

    @section api Documentation

    - @ref liblbfgs_api "libLBFGS API"

    @section sample Sample code

    @include sample.c

    @section ack Acknowledgements

    The L-BFGS algorithm is described in:
        - Jorge Nocedal.
          Updating Quasi-Newton Matrices with Limited Storage.
          <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
        - Dong C. Liu and Jorge Nocedal.
          On the limited memory BFGS method for large scale optimization.
          <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

    The line search algorithms used in this implementation are described in:
        - John E. Dennis and Robert B. Schnabel.
          <i>Numerical Methods for Unconstrained Optimization and Nonlinear
          Equations</i>, Englewood Cliffs, 1983.
        - Jorge J. More and David J. Thuente.
          Line search algorithm with guaranteed sufficient decrease.
          <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
          pp. 286-307, 1994.

    This library also implements Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
    method presented in:
        - Galen Andrew and Jianfeng Gao.
          Scalable training of L1-regularized log-linear models.
          In <i>Proceedings of the 24th International Conference on Machine
          Learning (ICML 2007)</i>, pp. 33-40, 2007.

    Special thanks go to:
        - Yoshimasa Tsuruoka and Daisuke Okanohara for technical information about
          OWL-QN
        - Takashi Imamichi for the useful enhancements of the backtracking method
        - Kevin S. Van Horn, Nic Schraudolph, and Tamas Nepusz for bug fixes

    Finally I would like to thank the original author, Jorge Nocedal, who has been
    distributing the effieicnt and explanatory implementation in an open source
    licence.

    @section reference Reference

    - <a href="http://www.ece.northwestern.edu/~nocedal/lbfgs.html">L-BFGS</a> by Jorge Nocedal.
    - <a href="http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/default.aspx">Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives</a> by Galen Andrew.
    - <a href="http://chasen.org/~taku/software/misc/lbfgs/">C port (via f2c)</a> by Taku Kudo.
    - <a href="http://www.alglib.net/optimization/lbfgs.php">C#/C++/Delphi/VisualBasic6 port</a> in ALGLIB.
    - <a href="http://cctbx.sourceforge.net/">Computational Crystallography Toolbox</a> includes
      <a href="http://cctbx.sourceforge.net/current_cvs/c_plus_plus/namespacescitbx_1_1lbfgs.html">scitbx::lbfgs</a>.
    */

    // #endif/*__LBFGS_H__*/

    
    
    
    
    
    
    
    /*
     *      Limited memory BFGS (L-BFGS).
     *
     * Copyright (c) 1990, Jorge Nocedal
     * Copyright (c) 2007-2010 Naoaki Okazaki
     * All rights reserved.
     *
     * Permission is hereby granted, free of charge, to any person obtaining a copy
     * of this software and associated documentation files (the "Software"), to deal
     * in the Software without restriction, including without limitation the rights
     * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
     * copies of the Software, and to permit persons to whom the Software is
     * furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in
     * all copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
     * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
     * THE SOFTWARE.
     */

    /* $Id$ */

    /*
    This library is a C port of the FORTRAN implementation of Limited-memory
    Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) method written by Jorge Nocedal.
    The original FORTRAN source code is available at:
    http://www.ece.northwestern.edu/~nocedal/lbfgs.html

    The L-BFGS algorithm is described in:
        - Jorge Nocedal.
          Updating Quasi-Newton Matrices with Limited Storage.
          <i>Mathematics of Computation</i>, Vol. 35, No. 151, pp. 773--782, 1980.
        - Dong C. Liu and Jorge Nocedal.
          On the limited memory BFGS method for large scale optimization.
          <i>Mathematical Programming</i> B, Vol. 45, No. 3, pp. 503-528, 1989.

    The line search algorithms used in this implementation are described in:
        - John E. Dennis and Robert B. Schnabel.
          <i>Numerical Methods for Unconstrained Optimization and Nonlinear
          Equations</i>, Englewood Cliffs, 1983.
        - Jorge J. More and David J. Thuente.
          Line search algorithm with guaranteed sufficient decrease.
          <i>ACM Transactions on Mathematical Software (TOMS)</i>, Vol. 20, No. 3,
          pp. 286-307, 1994.

    This library also implements Orthant-Wise Limited-memory Quasi-Newton (OWL-QN)
    method presented in:
        - Galen Andrew and Jianfeng Gao.
          Scalable training of L1-regularized log-linear models.
          In <i>Proceedings of the 24th International Conference on Machine
          Learning (ICML 2007)</i>, pp. 33-40, 2007.

    I would like to thank the original author, Jorge Nocedal, who has been
    distributing the effieicnt and explanatory implementation in an open source
    licence.
    */

    private static double min2(double a, double b) { return      ((a) <= (b) ? (a) : (b)); }
    private static double max2(double a, double b) { return      ((a) >= (b) ? (a) : (b)); }
    private static double max3(double a, double b, double c) {  return max2(max2((a), (b)), (c)); }

    private static class iteration_data_t {
        double alpha;
        double[] s;     /* [n] */
        double[] y;     /* [n] */
        double ys;     /* vecdot(y, s) */
    }
    
    private enum LineSearchAlgInternal {
        morethuente,
        backtracking_owlqn,
        backtracking,
    }
    
    /**
     * Start a L-BFGS optimization.
     *
     *  @param  n           The number of variables.
     *  @param  x           The array of variables. A client program can set
     *                      default values for the optimization and receive the
     *                      optimization result through this array. This array
     *                      must be allocated by ::lbfgs_malloc function
     *                      for libLBFGS built with SSE/SSE2 optimization routine
     *                      enabled. The library built without SSE/SSE2
     *                      optimization does not have such a requirement.
     *  @param  ptr_fx      The pointer to the variable that receives the final
     *                      value of the objective function for the variables.
     *                      This argument can be set to \c null if the final
     *                      value of the objective function is unnecessary.
     *  @param  proc_evaluate   The callback function to provide function and
     *                          gradient evaluations given a current values of
     *                          variables. A client program must implement a
     *                          callback function compatible with \ref
     *                          lbfgs_evaluate_t and pass the pointer to the
     *                          callback function.
     *  @param  proc_progress   The callback function to receive the progress
     *                          (the number of iterations, the current value of
     *                          the objective function) of the minimization
     *                          process. This argument can be set to \c null if
     *                          a progress report is unnecessary.
     *  @param  instance    A user data for the client program. The callback
     *                      functions will receive the value of this argument.
     *  @param  param       The pointer to a structure representing parameters for
     *                      L-BFGS optimization. A client program can set this
     *                      parameter to \c null to use the default parameters.
     *                      Call lbfgs_parameter_init() function to fill a
     *                      structure with the default values.
     *  @retval int         The status code. This function returns zero if the
     *                      minimization process terminates without an error. A
     *                      non-zero value indicates an error.
     */
    public static StatusCode lbfgs(
        double[] x,
        MutableDouble ptr_fx,
        callback_data_t cd,
        lbfgs_parameter_t _param
        )     
    {
        int n = cd.n;
        StatusCode ret, ls_ret;
        MutableInt ls = new MutableInt(0);
        int i, j, k, end, bound;        
        double step;
    
        /* Constant parameters and their default values. */
        lbfgs_parameter_t param = (_param != null) ? _param : new lbfgs_parameter_t();
        final int m = param.m;

        double[] xp;
        double[] g, gp, pg = null;
        double[] d, w, pf = null;
        iteration_data_t[] lm = null;
        iteration_data_t it = null;
        double ys, yy;
        double xnorm, gnorm, beta;
        double fx = 0.;
        double rate = 0.;
        LineSearchAlgInternal linesearch_choice = LineSearchAlgInternal.morethuente;

        /* Check the input parameters for errors. */
        if (n <= 0) {
            return LBFGSERR_INVALID_N;
        }
        if (param.epsilon < 0.) {
            return LBFGSERR_INVALID_EPSILON;
        }
        if (param.past < 0) {
            return LBFGSERR_INVALID_TESTPERIOD;
        }
        if (param.delta < 0.) {
            return LBFGSERR_INVALID_DELTA;
        }
        if (param.min_step < 0.) {
            return LBFGSERR_INVALID_MINSTEP;
        }
        if (param.max_step < param.min_step) {
            return LBFGSERR_INVALID_MAXSTEP;
        }
        if (param.ftol < 0.) {
            return LBFGSERR_INVALID_FTOL;
        }
        if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE ||
            param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
            if (param.wolfe <= param.ftol || 1. <= param.wolfe) {
                return LBFGSERR_INVALID_WOLFE;
            }
        }
        if (param.gtol < 0.) {
            return LBFGSERR_INVALID_GTOL;
        }
        if (param.xtol < 0.) {
            return LBFGSERR_INVALID_XTOL;
        }
        if (param.max_linesearch <= 0) {
            return LBFGSERR_INVALID_MAXLINESEARCH;
        }
        if (param.orthantwise_c < 0.) {
            return LBFGSERR_INVALID_ORTHANTWISE;
        }
        if (param.orthantwise_start < 0 || n < param.orthantwise_start) {
            return LBFGSERR_INVALID_ORTHANTWISE_START;
        }
        if (param.orthantwise_end < 0) {
            param.orthantwise_end = n;
        }
        if (n < param.orthantwise_end) {
            return LBFGSERR_INVALID_ORTHANTWISE_END;
        }
        if (param.orthantwise_c != 0.) {
            switch (param.linesearch) {
            case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
                linesearch_choice = LineSearchAlgInternal.backtracking_owlqn;
                break;
            default:
                /* Only the backtracking method is available. */
                return LBFGSERR_INVALID_LINESEARCH;
            }
        } else {
            switch (param.linesearch) {
            case LBFGS_LINESEARCH_MORETHUENTE:
                linesearch_choice = LineSearchAlgInternal.morethuente;
                break;
            case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
            case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
            case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
                linesearch_choice = LineSearchAlgInternal.backtracking;
                break;
            default:
                return LBFGSERR_INVALID_LINESEARCH;
            }
        }

        /* Allocate working space. */
        xp = new double[n];
        g = new double[n];
        gp = new double[n];
        d = new double[n];
        w = new double[n];

        if (param.orthantwise_c != 0.) {
            /* Allocate working space for OW-LQN. */
            pg = new double[n];
        }

        /* Allocate limited memory storage. */
        lm = new iteration_data_t[m];

        /* Initialize the limited memory. */
        for (i = 0;i < m;++i) {
            lm[i] = new iteration_data_t();
            it = lm[i];
            it.alpha = 0;
            it.ys = 0;
            it.s = new double[n];
            it.y = new double[n];
        }

        /* Allocate an array for storing previous values of the objective function. */
        if (0 < param.past) {
            pf = new double[param.past];
        }

        /* Evaluate the function value and its gradient. */
        fx = cd.proc_evaluate(cd.instance, x, g, cd.n, 0);
        if (0. != param.orthantwise_c) {
            /* Compute the L1 norm of the variable and add it to the object value. */
            xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
            fx += xnorm * param.orthantwise_c;
            owlqn_pseudo_gradient(
                pg, x, g, n,
                param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
                );
        }

        /* Store the initial value of the objective function. */
        if (pf != null) {
            pf[0] = fx;
        }

        /*
            Compute the direction;
            we assume the initial hessian matrix H_0 as the identity matrix.
         */
        if (param.orthantwise_c == 0.) {
            vecncpy(d, g, n);
        } else {
            vecncpy(d, pg, n);
        }

        /*
           Make sure that the initial variables are not a minimizer.
         */
        xnorm = vec2norm(x, n);
        if (param.orthantwise_c == 0.) {
            gnorm = vec2norm(g, n);
        } else {
            gnorm = vec2norm(pg, n);
        }
        if (xnorm < 1.0) xnorm = 1.0;
        if (gnorm / xnorm <= param.epsilon) {
            ptr_fx.v = fx;
            return LBFGS_ALREADY_MINIMIZED;
        }

        /* Compute the initial step:
            step = 1.0 / sqrt(vecdot(d, d, n))
         */
        step = vec2norminv(d, n);

        k = 1;
        end = 0;
        for (;;) {
            /* Store the current position and gradient vectors. */
            veccpy(xp, x, n);
            veccpy(gp, g, n);

            /* Search for an optimal step. */
            MutableDouble fxRef = new MutableDouble(fx);
            MutableDouble stepRef = new MutableDouble(step);
            if (param.orthantwise_c == 0.) {
                ls_ret = linesearch(n, x, fxRef, g, d, stepRef, xp, gp, w, cd, param, ls, linesearch_choice);
            } else {
                ls_ret = linesearch(n, x, fxRef, g, d, stepRef, xp, pg, w, cd, param, ls, linesearch_choice);
                owlqn_pseudo_gradient(
                    pg, x, g, n,
                    param.orthantwise_c, param.orthantwise_start, param.orthantwise_end
                    );
            }
            fx = fxRef.v;
            step = stepRef.v;
            
            if (ls_ret.ret < 0) {
                /* Revert to the previous point. */
                veccpy(x, xp, n);
                veccpy(g, gp, n);
                ptr_fx.v = fx;
                return ls_ret;
            }

            /* Compute x and g norms. */
            xnorm = vec2norm(x, n);
            if (param.orthantwise_c == 0.) {
                gnorm = vec2norm(g, n);
            } else {
                gnorm = vec2norm(pg, n);
            }

            /* Report the progress. */
            ret = cd.proc_progress(cd.instance, x, g, fx, xnorm, gnorm, step, cd.n, k, ls.v);
            if (ret.ret != 0) {
                ptr_fx.v = fx;
                return ret;
            }

            /*
                Convergence test.
                The criterion is given by the following formula:
                    |g(x)| / \max(1, |x|) < \epsilon
             */
            if (xnorm < 1.0) xnorm = 1.0;
            if (gnorm / xnorm <= param.epsilon) {
                /* Convergence. */
                ret = LBFGS_SUCCESS;
                break;
            }

            /*
                Test for stopping criterion.
                The criterion is given by the following formula:
                    (f(past_x) - f(x)) / f(x) < \delta
             */
            if (pf != null) {
                /* We don't test the stopping criterion while k < past. */
                if (param.past <= k) {
                    /* Compute the relative improvement from the past. */
                    rate = (pf[k % param.past] - fx) / fx;

                    /* The stopping criterion. */
                    if (rate < param.delta) {
                        ret = LBFGS_STOP;
                        break;
                    }
                }

                /* Store the current value of the objective function. */
                pf[k % param.past] = fx;
            }

            if (param.max_iterations != 0 && param.max_iterations < k+1) {
                /* Maximum number of iterations. */
                ret = LBFGSERR_MAXIMUMITERATION;
                break;
            }

            /*
                Update vectors s and y:
                    s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
                    y_{k+1} = g_{k+1} - g_{k}.
             */
            it = lm[end];
            vecdiff(it.s, x, xp, n);
            vecdiff(it.y, g, gp, n);

            /*
                Compute scalars ys and yy:
                    ys = y^t \cdot s = 1 / \rho.
                    yy = y^t \cdot y.
                Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
             */
            ys = vecdot(it.y, it.s, n);
            yy = vecdot(it.y, it.y, n);
            it.ys = ys;

            /*
                Recursive formula to compute dir = -(H \cdot g).
                    This is described in page 779 of:
                    Jorge Nocedal.
                    Updating Quasi-Newton Matrices with Limited Storage.
                    Mathematics of Computation, Vol. 35, No. 151,
                    pp. 773--782, 1980.
             */
            bound = (m <= k) ? m : k;
            ++k;
            end = (end + 1) % m;

            /* Compute the steepest direction. */
            if (param.orthantwise_c == 0.) {
                /* Compute the negative of gradients. */
                vecncpy(d, g, n);
            } else {
                vecncpy(d, pg, n);
            }

            j = end;
            for (i = 0;i < bound;++i) {
                j = (j + m - 1) % m;    /* if (--j == -1) j = m-1; */
                it = lm[j];
                /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
                it.alpha = vecdot(it.s, d, n);
                it.alpha /= it.ys;
                /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
                vecadd(d, it.y, -it.alpha, n);
            }

            vecscale(d, ys / yy, n);

            for (i = 0;i < bound;++i) {
                it = lm[j];
                /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
                beta = vecdot(it.y, d, n);
                beta /= it.ys;
                /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
                vecadd(d, it.s, it.alpha - beta, n);
                j = (j + 1) % m;        /* if (++j == m) j = 0; */
            }

            /*
                Constrain the search direction for orthant-wise updates.
             */
            if (param.orthantwise_c != 0.) {
                for (i = param.orthantwise_start;i < param.orthantwise_end;++i) {
                    if (d[i] * pg[i] >= 0) {
                        d[i] = 0;
                    }
                }
            }

            /*
                Now the search direction d is ready. We try step = 1 first.
             */
            step = 1.0;
        }

        /* Return the final value of the objective function. */
        ptr_fx.v = fx;
        
        return ret;
    }

    private static StatusCode linesearch(
        int n,
        double[] x,
        MutableDouble f,
        double[] g,
        double[] s,
        MutableDouble stp,
        final double[] xp,
        final double[] gp,
        double[] wp,
        callback_data_t cd,
        final lbfgs_parameter_t param,
        MutableInt ls,
        LineSearchAlgInternal linesearch_choice
        )
    {
        switch(linesearch_choice) {
        case backtracking: return line_search_backtracking(n, x, f, g, s, stp, xp, gp, wp, cd, param, ls);
        case backtracking_owlqn: return line_search_backtracking_owlqn(n, x, f, g, s, stp, xp, gp, wp, cd, param, ls);
        case morethuente: return line_search_morethuente(n, x, f, g, s, stp, xp, gp, wp, cd, param, ls);
        default: throw new RuntimeException("line search not yet implemented: " + linesearch_choice); 
        }
    }


    static StatusCode line_search_backtracking(
        int n,
        double[] x,
        MutableDouble f,
        double[] g,
        double[] s,
        MutableDouble stp,
        final double[] xp,
        final double[] gp,
        double[] wp,
        callback_data_t cd,
        final lbfgs_parameter_t param,
        MutableInt ls
        )
    {
        int count = 0;
        double width, dg;
        double finit, dginit = 0., dgtest;
        final double dec = 0.5, inc = 2.1;

        /* Check the input parameters for errors. */
        if (stp.v <= 0.) {
            return LBFGSERR_INVALIDPARAMETERS;
        }

        /* Compute the initial gradient in the search direction. */
        dginit = vecdot(g, s, n);

        /* Make sure that s points to a descent direction. */
        if (0 < dginit) {
            return LBFGSERR_INCREASEGRADIENT;
        }

        /* The initial value of the objective function. */
        finit = f.v;
        dgtest = param.ftol * dginit;

        for (;;) {
            veccpy(x, xp, n);
            vecadd(x, s, stp.v, n);

            /* Evaluate the function and gradient values. */
            f.v = cd.proc_evaluate(cd.instance, x, g, cd.n, stp.v);

            ++count;

            if (f.v > finit + stp.v * dgtest) {
                width = dec;
            } else {
                /* The sufficient decrease condition (Armijo condition). */
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
                    /* Exit with the Armijo condition. */
                    ls.v = count;
                    return LBFGS_LS_SUCCESS;
                }

                /* Check the Wolfe condition. */
                dg = vecdot(g, s, n);
                if (dg < param.wolfe * dginit) {
                    width = inc;
                } else {
                    if(param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE) {
                        /* Exit with the regular Wolfe condition. */
                        ls.v = count;
                        return LBFGS_LS_SUCCESS;
                    }

                    /* Check the strong Wolfe condition. */
                    if(dg > -param.wolfe * dginit) {
                        width = dec;
                    } else {
                        /* Exit with the strong Wolfe condition. */
                        ls.v = count;
                        return LBFGS_LS_SUCCESS;
                    }
                }
            }

            if (stp.v < param.min_step) {
                /* The step is the minimum value. */
                return LBFGSERR_MINIMUMSTEP;
            }
            if (stp.v > param.max_step) {
                /* The step is the maximum value. */
                return LBFGSERR_MAXIMUMSTEP;
            }
            if (param.max_linesearch <= count) {
                /* Maximum number of iteration. */
                return LBFGSERR_MAXIMUMLINESEARCH;
            }

            stp.v *= width;
        }
    }



    static StatusCode line_search_backtracking_owlqn(
        int n,
        double[] x,
        MutableDouble f,
        double[] g,
        double[] s,
        MutableDouble stp,
        final double[] xp,
        final double[] gp,
        double[] wp,
        callback_data_t cd,
        final lbfgs_parameter_t param,
        MutableInt ls
        )
    {
        int i, count = 0;
        double width = 0.5, norm = 0.;
        double finit = f.v, dgtest;

        /* Check the input parameters for errors. */
        if (stp.v <= 0.) {
            return LBFGSERR_INVALIDPARAMETERS;
        }

        /* Choose the orthant for the new point. */
        for (i = 0;i < n;++i) {
            wp[i] = (xp[i] == 0.) ? -gp[i] : xp[i];
        }

        for (;;) {
            /* Update the current point. */
            veccpy(x, xp, n);
            vecadd(x, s, stp.v, n);

            /* The current point is projected onto the orthant. */
            owlqn_project(x, wp, param.orthantwise_start, param.orthantwise_end);

            /* Evaluate the function and gradient values. */
            f.v = cd.proc_evaluate(cd.instance, x, g, cd.n, stp.v);

            /* Compute the L1 norm of the variables and add it to the object value. */
            norm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
            f.v += norm * param.orthantwise_c;

            ++count;

            dgtest = 0.;
            for (i = 0;i < n;++i) {
                dgtest += (x[i] - xp[i]) * gp[i];
            }

            if (f.v <= finit + param.ftol * dgtest) {
                /* The sufficient decrease condition. */
                ls.v = count;
                return LBFGS_LS_SUCCESS;
            }

            if (stp.v < param.min_step) {
                /* The step is the minimum value. */
                return LBFGSERR_MINIMUMSTEP;
            }
            if (stp.v > param.max_step) {
                /* The step is the maximum value. */
                return LBFGSERR_MAXIMUMSTEP;
            }
            if (param.max_linesearch <= count) {
                /* Maximum number of iteration. */
                return LBFGSERR_MAXIMUMLINESEARCH;
            }

            stp.v *= width;
        }
    }

    static double owlqn_x1norm(
        final double[] x,
        final int start,
        final int n
        )
    {
        int i;
        double norm = 0.;

        for (i = start;i < n;++i) {
            norm += Math.abs(x[i]);
        }

        return norm;
    }

    static void owlqn_pseudo_gradient(
        double[] pg,
        final double[] x,
        final double[] g,
        final int n,
        final double c,
        final int start,
        final int end
        )
    {
        int i;

        /* Compute the negative of gradients. */
        for (i = 0;i < start;++i) {
            pg[i] = g[i];
        }

        /* Compute the psuedo-gradients. */
        for (i = start;i < end;++i) {
            if (x[i] < 0.) {
                /* Differentiable. */
                pg[i] = g[i] - c;
            } else if (0. < x[i]) {
                /* Differentiable. */
                pg[i] = g[i] + c;
            } else {
                if (g[i] < -c) {
                    /* Take the right partial derivative. */
                    pg[i] = g[i] + c;
                } else if (c < g[i]) {
                    /* Take the left partial derivative. */
                    pg[i] = g[i] - c;
                } else {
                    pg[i] = 0.;
                }
            }
        }

        for (i = end;i < n;++i) {
            pg[i] = g[i];
        }
    }

    static void owlqn_project(
        double[] d,
        final double[] sign,
        final int start,
        final int end
        )
    {
        int i;

        for (i = start;i < end;++i) {
            if (d[i] * sign[i] <= 0) {
                d[i] = 0;
            }
        }
    }

    /* ----------- Vector Arithmetic ------- */
    
    private static void vecset(double[]x, final double c, final int n)
    {
        int i;
        
        for (i = 0;i < n;++i) {
            x[i] = c;
        }
    }
    
    private static void veccpy(double[]y, final double[]x, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            y[i] = x[i];
        }
    }
    
    private static void vecncpy(double[]y, final double[]x, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            y[i] = -x[i];
        }
    }
    
    private static void vecadd(double[]y, final double[]x, final double c, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            y[i] += c * x[i];
        }
    }
    
    private static void vecdiff(double[]z, final double[]x, final double[]y, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            z[i] = x[i] - y[i];
        }
    }
    
    private static void vecscale(double[]y, final double c, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            y[i] *= c;
        }
    }
    
    private static void vecmul(double[]y, final double[]x, final int n)
    {
        int i;
    
        for (i = 0;i < n;++i) {
            y[i] *= x[i];
        }
    }
    
    private static double vecdot(final double[]x, final double[]y, final int n)
    {
        int i;
        double s = 0.;
        for (i = 0;i < n;++i) {
            s += x[i] * y[i];
        }
        return s;
    }
    
    private static double vec2norm(final double[]x, final int n)
    {
        double s = vecdot(x, x, n);
        return Math.sqrt(s);
    }
    
    private static double vec2norminv(final double[]x, final int n)
    {
        double s = vec2norm(x, n);
        return (1.0 / s);
    }
    
    private static boolean fsigndiff(double x, double y) {
        return (x * y / Math.abs(y)) < 0.;   
    }

    
    

    static StatusCode line_search_morethuente(
        int n,
        double[] x,
        MutableDouble f,
        double[] g,
        double[] s,
        MutableDouble stp,
        final double[] xp,
        final double[] gp,
        double[] wp,
        callback_data_t cd,
        final lbfgs_parameter_t param,
        MutableInt ls
        )
    {
        int count = 0;
        int brackt, stage1;
        StatusCode uinfo = LBFGS_CONTINUE;
        double dg;
        double stx, fx, dgx;
        double sty, fy, dgy;
        double fxm, dgxm, fym, dgym, fm, dgm;
        double finit, ftest1, dginit, dgtest;
        double width, prev_width;
        double stmin, stmax;

        /* Check the input parameters for errors. */
        if (stp.v <= 0.) {
            return LBFGSERR_INVALIDPARAMETERS;
        }

        /* Compute the initial gradient in the search direction. */
        dginit = vecdot(g, s, n);

        /* Make sure that s points to a descent direction. */
        if (0 < dginit) {
            return LBFGSERR_INCREASEGRADIENT;
        }

        /* Initialize local variables. */
        brackt = 0;
        stage1 = 1;
        finit = f.v;
        dgtest = param.ftol * dginit;
        width = param.max_step - param.min_step;
        prev_width = 2.0 * width;

        /*
            The variables stx, fx, dgx contain the values of the step,
            function, and directional derivative at the best step.
            The variables sty, fy, dgy contain the value of the step,
            function, and derivative at the other endpoint of
            the interval of uncertainty.
            The variables stp, f, dg contain the values of the step,
            function, and derivative at the current step.
        */
        stx = sty = 0.;
        fx = fy = finit;
        dgx = dgy = dginit;

        for (;;) {
            /*
                Set the minimum and maximum steps to correspond to the
                present interval of uncertainty.
             */
            if (brackt != 0) {
                stmin = min2(stx, sty);
                stmax = max2(stx, sty);
            } else {
                stmin = stx;
                stmax = stp.v + 4.0 * (stp.v - stx);
            }

            /* Clip the step in the range of [stpmin, stpmax]. */
            if (stp.v < param.min_step) stp.v = param.min_step;
            if (param.max_step < stp.v) stp.v = param.max_step;

            /*
                If an unusual termination is to occur then let
                stp be the lowest point obtained so far.
             */
            if ((brackt != 0 && ((stp.v <= stmin || stmax <= stp.v) || param.max_linesearch <= count + 1 || uinfo.ret != 0)) 
                    || (brackt != 0 && (stmax - stmin <= param.xtol * stmax))) {
                stp.v = stx;
            }

            /*
                Compute the current value of x:
                    x <- x + (*stp) * s.
             */
            veccpy(x, xp, n);
            vecadd(x, s, stp.v, n);

            /* Evaluate the function and gradient values. */
            f.v = cd.proc_evaluate(cd.instance, x, g, cd.n, stp.v);
            dg = vecdot(g, s, n);

            ftest1 = finit + stp.v * dgtest;
            ++count;

            /* Test for errors and convergence. */
            if (brackt != 0 && ((stp.v <= stmin || stmax <= stp.v) || uinfo.ret != 0)) {
                /* Rounding errors prevent further progress. */
                return LBFGSERR_ROUNDING_ERROR;
            }
            if (stp.v == param.max_step && f.v <= ftest1 && dg <= dgtest) {
                /* The step is the maximum value. */
                return LBFGSERR_MAXIMUMSTEP;
            }
            if (stp.v == param.min_step && (ftest1 < f.v || dgtest <= dg)) {
                /* The step is the minimum value. */
                return LBFGSERR_MINIMUMSTEP;
            }
            if (brackt != 0 && (stmax - stmin) <= param.xtol * stmax) {
                /* Relative width of the interval of uncertainty is at most xtol. */
                return LBFGSERR_WIDTHTOOSMALL;
            }
            if (param.max_linesearch <= count) {
                /* Maximum number of iteration. */
                return LBFGSERR_MAXIMUMLINESEARCH;
            }
            if (f.v <= ftest1 && Math.abs(dg) <= param.gtol * (-dginit)) {
                /* The sufficient decrease condition and the directional derivative condition hold. */
                ls.v = count;
                return LBFGS_LS_SUCCESS;
            }

            /*
                In the first stage we seek a step for which the modified
                function has a nonpositive value and nonnegative derivative.
             */
            if (stage1 != 0 && f.v <= ftest1 && min2(param.ftol, param.gtol) * dginit <= dg) {
                stage1 = 0;
            }

            /*
                A modified function is used to predict the step only if
                we have not obtained a step for which the modified
                function has a nonpositive function value and nonnegative
                derivative, and if a lower function value has been
                obtained but the decrease is not sufficient.
             */
            if (stage1 != 0 && ftest1 < f.v && f.v <= fx) {
                /* Define the modified function and derivative values. */
                fm = f.v - stp.v * dgtest;
                fxm = fx - stx * dgtest;
                fym = fy - sty * dgtest;
                dgm = dg - dgtest;
                dgxm = dgx - dgtest;
                dgym = dgy - dgtest;

                /*
                    Call update_trial_interval() to update the interval of
                    uncertainty and to compute the new step.
                 */
                UpdVals uv = new UpdVals(
                        stx, fxm, dgxm,
                        sty, fym, dgym,
                        stp.v, fm, dgm,
                        stmin, stmax, brackt);
                uinfo = update_trial_interval(uv);
                stx = uv.x;
                fxm = uv.fx;
                dgxm = uv.dx;
                sty = uv.y;
                fym = uv.fy;
                dgym = uv.dy;
                stp.v = uv.t;
                fm = uv.ft;
                dgm = uv.dt;
                stmin = uv.tmin;
                stmax = uv.tmax;
                brackt = uv.brackt;

                /* Reset the function and gradient values for f. */
                fx = fxm + stx * dgtest;
                fy = fym + sty * dgtest;
                dgx = dgxm + dgtest;
                dgy = dgym + dgtest;
            } else {
                /*
                    Call update_trial_interval() to update the interval of
                    uncertainty and to compute the new step.
                 */
                UpdVals uv = new UpdVals(
                        stx, fx, dgx,
                        sty, fy, dgy,
                        stp.v, f.v, dg,
                        stmin, stmax, brackt);
                uinfo = update_trial_interval(uv);
                stx = uv.x;
                fx = uv.fx;
                dgx = uv.dx;
                sty = uv.y;
                fy = uv.fy;
                dgy = uv.dy;
                stp.v = uv.t;
                f.v = uv.ft;
                dg = uv.dt;
                stmin = uv.tmin;
                stmax = uv.tmax;
                brackt = uv.brackt;
            }

            /*
                Force a sufficient decrease in the interval of uncertainty.
             */
            if (brackt != 0) {
                if (0.66 * prev_width <= Math.abs(sty - stx)) {
                    stp.v = stx + 0.5 * (sty - stx);
                }
                prev_width = width;
                width = Math.abs(sty - stx);
            }
        }

        // Unreachable.
        //return LBFGSERR_LOGICERROR;
    }
    
    /**
     * Find a minimizer of an interpolated cubic function.
     *  @param  cm      The minimizer of the interpolated cubic.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     *  @param  du      The value of f'(v).
     */
    private static double CUBIC_MINIMIZER(double cm, double u, double fu, double du, double v, double fv, double dv) {
        double a, d, gamma, theta, p, q, r, s;
        d = (v) - (u);
        theta = ((fu) - (fv)) * 3 / d + (du) + (dv);
        p = Math.abs(theta);
        q = Math.abs(du);
        r = Math.abs(dv);
        s = max3(p, q, r);
        /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
        a = theta / s;
        gamma = s * Math.sqrt(a * a - ((du) / s) * ((dv) / s));
        if ((v) < (u)) gamma = -gamma;
        p = gamma - (du) + theta;
        q = gamma - (du) + gamma + (dv);
        r = p / q;
        (cm) = (u) + r * d;
        return cm;
    }

    /**
     * Find a minimizer of an interpolated cubic function.
     *  @param  cm      The minimizer of the interpolated cubic.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     *  @param  du      The value of f'(v).
     *  @param  xmin    The maximum value.
     *  @param  xmin    The minimum value.
     */
    private static double CUBIC_MINIMIZER2(double cm, double u, double fu, double du, double v, double fv, double dv, double xmin, double xmax) {
        double a, d, gamma, theta, p, q, r, s;
        d = (v) - (u);
        theta = ((fu) - (fv)) * 3 / d + (du) + (dv);
        p = Math.abs(theta);
        q = Math.abs(du);
        r = Math.abs(dv);
        s = max3(p, q, r);
        /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
        a = theta / s;
        gamma = s * Math.sqrt(max2(0, a * a - ((du) / s) * ((dv) / s)));
        if ((u) < (v)) gamma = -gamma;
        p = gamma - (dv) + theta;
        q = gamma - (dv) + gamma + (du);
        r = p / q;
        if (r < 0. && gamma != 0.) {
            (cm) = (v) - r * d;
        } else if (a < 0) {
            (cm) = (xmax);
        } else {
            (cm) = (xmin);
        }
        return cm;
    }

    /**
     * Find a minimizer of an interpolated quadratic function.
     *  @param  qm      The minimizer of the interpolated quadratic.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     */
    private static double QUARD_MINIMIZER(double qm, double u, double fu, double du, double v, double fv) {
        double a, d, gamma, theta, p, q, r, s;
        a = (v) - (u);
        (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;
        return qm;
    }

    /**
     * Find a minimizer of an interpolated quadratic function.
     *  @param  qm      The minimizer of the interpolated quadratic.
     *  @param  u       The value of one point, u.
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  dv      The value of f'(v).
     */
    private static double QUARD_MINIMIZER2(double qm, double u, double du, double v, double dv) {
        double a, d, gamma, theta, p, q, r, s;
        a = (u) - (v);
        (qm) = (v) + (dv) / ((dv) - (du)) * a;
        return qm;
    }

    private static class UpdVals {
        double x;
        double fx;
        double dx;
        double y;
        double fy;
        double dy;
        double t;
        double ft;
        double dt;
        final double tmin;
        final double tmax;
        int brackt;

        public UpdVals(double x, double fx, double dx, double y, double fy, double dy, double t, double ft, double dt,
                double tmin, double tmax, int brackt) {
            super();
            this.x = x;
            this.fx = fx;
            this.dx = dx;
            this.y = y;
            this.fy = fy;
            this.dy = dy;
            this.t = t;
            this.ft = ft;
            this.dt = dt;
            this.tmin = tmin;
            this.tmax = tmax;
            this.brackt = brackt;
        }

    }
    

    /**
     * Update a safeguarded trial value and interval for line search.
     *
     *  The parameter x represents the step with the least function value.
     *  The parameter t represents the current step. This function assumes
     *  that the derivative at the point of x in the direction of the step.
     *  If the bracket is set to true, the minimizer has been bracketed in
     *  an interval of uncertainty with endpoints between x and y.
     *
     *  @param  x       The pointer to the value of one endpoint.
     *  @param  fx      The pointer to the value of f(x).
     *  @param  dx      The pointer to the value of f'(x).
     *  @param  y       The pointer to the value of another endpoint.
     *  @param  fy      The pointer to the value of f(y).
     *  @param  dy      The pointer to the value of f'(y).
     *  @param  t       The pointer to the value of the trial value, t.
     *  @param  ft      The pointer to the value of f(t).
     *  @param  dt      The pointer to the value of f'(t).
     *  @param  tmin    The minimum value for the trial value, t.
     *  @param  tmax    The maximum value for the trial value, t.
     *  @param  brackt  The pointer to the predicate if the trial value is
     *                  bracketed.
     *  @retval int     Status value. Zero indicates a normal termination.
     *  
     *  @see
     *      Jorge J. More and David J. Thuente. Line search algorithm with
     *      guaranteed sufficient decrease. ACM Transactions on Mathematical
     *      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
     */
    // Old parameter order:
    // 
    //  double *x,
    //  double *fx,
    //  double *dx,
    //  double *y,
    //  double *fy,
    //  double *dy,
    //  double *t,
    //  double *ft,
    //  double *dt,
    //  const double tmin,
    //  const double tmax,
    //  int *brackt
    private static StatusCode update_trial_interval(UpdVals uv)
    {
        double x = uv.x;
        double fx = uv.fx;
        double dx = uv.dx;
        double y = uv.y;
        double fy = uv.fy;
        double dy = uv.dy;
        double t = uv.t;
        double ft = uv.ft;
        double dt = uv.dt;
        double tmin = uv.tmin;
        double tmax = uv.tmax;
        int brackt = uv.brackt;
        
        int bound;
        boolean dsign = fsigndiff(dt, dx);
        double mc = 0; /* minimizer of an interpolated cubic. */
        double mq = 0; /* minimizer of an interpolated quadratic. */
        double newt = 0;   /* new trial value. */
        // TODO: Remove: USES_MINIMIZER;     /* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

        /* Check the input parameters for errors. */
        if (brackt != 0) {
            if (t <= min2(x, y) || max2(x, y) <= t) {
                /* The trival value t is out of the interval. */
                return LBFGSERR_OUTOFINTERVAL;
            }
            if (0. <= dx * (t - x)) {
                /* The function must decrease from x. */
                return LBFGSERR_INCREASEGRADIENT;
            }
            if (tmax < tmin) {
                /* Incorrect tmin and tmax specified. */
                return LBFGSERR_INCORRECT_TMINMAX;
            }
        }

        /*
            Trial value selection.
         */
        if (fx < ft) {
            /*
                Case 1: a higher function value.
                The minimum is brackt. If the cubic minimizer is closer
                to x than the quadratic one, the cubic one is taken, else
                the average of the minimizers is taken.
             */
            brackt = 1;
            bound = 1;
            mc = CUBIC_MINIMIZER(mc, x, fx, dx, t, ft, dt);
            mq = QUARD_MINIMIZER(mq, x, fx, dx, t, ft);
            if (Math.abs(mc - x) < Math.abs(mq - x)) {
                newt = mc;
            } else {
                newt = mc + 0.5 * (mq - mc);
            }
        } else if (dsign) {
            /*
                Case 2: a lower function value and derivatives of
                opposite sign. The minimum is brackt. If the cubic
                minimizer is closer to x than the quadratic (secant) one,
                the cubic one is taken, else the quadratic one is taken.
             */
            brackt = 1;
            bound = 0;
            mc = CUBIC_MINIMIZER(mc, x, fx, dx, t, ft, dt);
            mq = QUARD_MINIMIZER2(mq, x, dx, t, dt);
            if (Math.abs(mc - t) > Math.abs(mq - t)) {
                newt = mc;
            } else {
                newt = mq;
            }
        } else if (Math.abs(dt) < Math.abs(dx)) {
            /*
                Case 3: a lower function value, derivatives of the
                same sign, and the magnitude of the derivative decreases.
                The cubic minimizer is only used if the cubic tends to
                infinity in the direction of the minimizer or if the minimum
                of the cubic is beyond t. Otherwise the cubic minimizer is
                defined to be either tmin or tmax. The quadratic (secant)
                minimizer is also computed and if the minimum is brackt
                then the the minimizer closest to x is taken, else the one
                farthest away is taken.
             */
            bound = 1;
            mc = CUBIC_MINIMIZER2(mc, x, fx, dx, t, ft, dt, tmin, tmax);
            mq = QUARD_MINIMIZER2(mq, x, dx, t, dt);
            if (brackt != 0) {
                if (Math.abs(t - mc) < Math.abs(t - mq)) {
                    newt = mc;
                } else {
                    newt = mq;
                }
            } else {
                if (Math.abs(t - mc) > Math.abs(t - mq)) {
                    newt = mc;
                } else {
                    newt = mq;
                }
            }
        } else {
            /*
                Case 4: a lower function value, derivatives of the
                same sign, and the magnitude of the derivative does
                not decrease. If the minimum is not brackt, the step
                is either tmin or tmax, else the cubic minimizer is taken.
             */
            bound = 0;
            if (brackt != 0) {
                newt = CUBIC_MINIMIZER(newt, t, ft, dt, y, fy, dy);
            } else if (x < t) {
                newt = tmax;
            } else {
                newt = tmin;
            }
        }

        /*
            Update the interval of uncertainty. This update does not
            depend on the new step or the case analysis above.

            - Case a: if f(x) < f(t),
                x <- x, y <- t.
            - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
                x <- t, y <- y.
            - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0, 
                x <- t, y <- x.
         */
        if (fx < ft) {
            /* Case a */
            y = t;
            fy = ft;
            dy = dt;
        } else {
            /* Case c */
            if (dsign) {
                y = x;
                fy = fx;
                dy = dx;
            }
            /* Cases b and c */
            x = t;
            fx = ft;
            dx = dt;
        }

        /* Clip the new trial value in [tmin, tmax]. */
        if (tmax < newt) newt = tmax;
        if (newt < tmin) newt = tmin;

        /*
            Redefine the new trial value if it is close to the upper bound
            of the interval.
         */
        if (brackt != 0 && bound != 0) {
            mq = x + 0.66 * (y - x);
            if (x < y) {
                if (mq < newt) newt = mq;
            } else {
                if (newt < mq) newt = mq;
            }
        }

        /* Return the new trial value. */
        t = newt;
        

        uv.x = x;
        uv.fx = fx;
        uv.dx = dx;
        uv.y = y;
        uv.fy = fy;
        uv.dy = dy;
        uv.t = t;
        uv.ft = ft;
        uv.dt = dt;
        //uv.tmin = tmin;
        //uv.tmax = tmax;
        uv.brackt = brackt;
        
        return LBFGS_CONTINUE;
    }


    
}


