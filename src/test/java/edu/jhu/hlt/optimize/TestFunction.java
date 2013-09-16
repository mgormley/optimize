package edu.jhu.hlt.optimize;

import static org.junit.Assert.*;

import org.junit.Test;

import junit.framework.Assert;

public class TestFunction {
	double value_at_optima;
	double value_tolerance;
	double [] param_at_optima;
	double param_tolerance;
	Function f;
	public TestFunction(Function f, double value_at_optima, double value_tolerance, double [] param_at_optima, double param_tolerance) {
		this.f = f;
		this.value_at_optima = value_at_optima;
		this.value_tolerance = value_tolerance;
		this.param_at_optima = param_at_optima;
		this.param_tolerance = param_tolerance;
	}
	
	public void checkValue(double value) {
		Assert.assertEquals(value, value_at_optima, value_tolerance);
	}
	
	public void checkParam(double [] param) {
		for(int i=0; i<param.length; i++) {
			Assert.assertEquals(param[i], param_at_optima[i], param_tolerance);
		}
	}
	
}
