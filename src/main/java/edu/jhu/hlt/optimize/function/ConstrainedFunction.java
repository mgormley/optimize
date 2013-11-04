package edu.jhu.hlt.optimize.function;

public interface ConstrainedFunction extends Function {
	public Bounds getBounds();
	public void setBounds(Bounds b);
}
