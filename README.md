# AdditiveSeparabilityTest
A test for additive separability of a function. Is the function of the form f(x,y) = g(x) + h(y)?

Many functions characterising physical systems are additively separable. This is the case, for instance, of mechanical Hamiltonian functions in physics, population growth equations in biology, and consumer preference and utility functions in economics. 
We consider the scenario in which a surrogate of a function is to be tested for additive separability. The detection that the surrogate is additively separable can be leveraged to improve further learning. 
Hence, it is beneficial to have the ability to test for such separability in surrogates. The mathematical approach is to test if the mixed partial derivative of the surrogate is zero; or empirically, lower than a threshold. We present and comparatively and empirically evaluate the eight methods to compute the mixed partial derivative of a surrogate function.


# How to use:
1. pip install the requirements.txt file
2. generate surrogates of functions that are either additively separable or not. check out the GenerateSurrogates folder. (you can skip this step if you already have a surrogate you want to test)
3. evaluate your surrogate. (Method 3 in the Classifiers folder works best if you have a time constraint)
