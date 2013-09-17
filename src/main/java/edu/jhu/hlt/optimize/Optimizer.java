package edu.jhu.hlt.optimize;

public abstract class Optimizer<T> {
	T f;
	public Optimizer(T f) {
		this.f = f;
	}
	public T getFunction() { return f; }
}
