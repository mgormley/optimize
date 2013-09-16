A Java optimization library


-------------------------------------------
Issues:
- Interface for Maximizer/Minimizer should not take an initial point. Instead that is part of the Function's state.
- All use of double[] should be replaced with IntDoubleVector interface.
- SGD/AdaGrad should implement Maximizer<DifferentiableFunction> and should have base classes for the batch cases.
- Framework for testing optimization methods.
- Example unit test for testing some function. (e.g. finite differences checks for functions.) 
- Example of how to use Apache library to compute gradient automatically.

-------------------------------------------
Wishlist:

