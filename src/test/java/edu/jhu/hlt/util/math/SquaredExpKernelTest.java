package edu.jhu.hlt.util.math;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.SGDQNCorrectedTest;

public class SquaredExpKernelTest {

	static Logger log = Logger.getLogger(SquaredExpKernelTest.class);
	
	@Test
	public void ADmethods() {
		
		BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		double [] a1 = new double [] {1.1, 2.4};
		double [] a2 = new double [] {0.5, 10.0};
		double [] a3 = new double [] {0.1, -5.0};
		
		DerivativeStructure [] a1_ad = new DerivativeStructure[2];
		DerivativeStructure [] a2_ad = new DerivativeStructure[2];
		DerivativeStructure [] a3_ad = new DerivativeStructure[2];
		
		RealVector a1_v = new ArrayRealVector(a1);
		RealVector a2_v = new ArrayRealVector(a2);
		RealVector a3_v = new ArrayRealVector(a3);
		
		for(int i=0; i<a1.length; i++) {
			a1_ad[i] = new DerivativeStructure(2, 1, i, a1[i]);
			a2_ad[i] = new DerivativeStructure(2, 1, i, a2[i]);
			a3_ad[i] = new DerivativeStructure(2, 1, i, a3[i]);
		}
		
		double variance = 1d;
		double len_scale = 0.1d;
		SquaredExpKernel kernel = new SquaredExpKernel();
		
		double res1 = kernel.k(a1_ad, a2_ad).getValue();
		double res2 = kernel.k(a1_v, a2_ad).getValue();
		double res3 = kernel.k(a1_v, a2_v);
		
	    assertEquals(res1, res2, 1e-3);
	    assertEquals(res2, res3, 1e-3);
	    
	    res1 = kernel.k(a2_ad, a3_ad).getValue();
	    res2 = kernel.k(a2_v, a3_ad).getValue();
	    res3 = kernel.k(a2_v, a3_v);
	    
	    log.info(res1);
	    log.info(res2);
	    log.info(res3);
	    
	    assertEquals(res1, res2, 1e-3);
	    assertEquals(res2, res3, 1e-3);
	}
	
	@Test
	public void squaredExpKernelTest() {
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		double [] a1 = new double [] {1,1};
		double [] a2 = new double [] {0,0};
		double [] a3 = new double [] {2,1};
		DerivativeStructure [] a2_ad = new DerivativeStructure[2];
		DerivativeStructure [] a3_ad = new DerivativeStructure[2];
		for(int i=0; i<a2.length; i++) {
			a2_ad[i] = new DerivativeStructure(2, 2, i, a2[i]);
			a3_ad[i] = new DerivativeStructure(2, 2, i, a3[i]);
		}
		RealVector x1 = new ArrayRealVector(a1);
		RealVector x2 = new ArrayRealVector(a2);
		RealVector x3 = new ArrayRealVector(a3);
		double variance = 1d;
		double len_scale = 1d;
		SquaredExpKernel kernel = new SquaredExpKernel(variance, len_scale);
		double d = kernel.k(x1, x2);
		double d2 = kernel.k(x2, x3);
		DerivativeStructure d_ad = kernel.k(x1, a2_ad);
		DerivativeStructure d2_ad = kernel.k(x2, a3_ad);
		log.info("d = " + d);
		log.info("d2 = " + d2);
		log.info("d_ad = " + d_ad.getValue());
		log.info("d2_ad = " + d2_ad.getValue());
		RealMatrix X = MatrixUtils.createRealMatrix(new double [][] {a1,a2,a3}).transpose();
		log.info("X rows = " + X.getRowDimension());
		log.info("X col = " + X.getColumnDimension());
		RealMatrix K = kernel.K(X);
		log.info("K = " + K);
	}
	
}
