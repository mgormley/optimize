package edu.jhu.hlt.optimize;

public interface ConstrainedFunction extends Function {
	public Bounds getBounds();
	public void setBounds(Bounds b);
}
