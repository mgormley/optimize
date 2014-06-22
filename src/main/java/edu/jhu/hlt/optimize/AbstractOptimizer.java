package edu.jhu.hlt.optimize;

// TODO: Remove this class. Optimizers should never be stateful.
@Deprecated
public abstract class AbstractOptimizer<T> {
	T f;
	public AbstractOptimizer(T f) {
		this.f = f;
	}
	public T getFunction() { return f; }
}
