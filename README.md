# Optimize [![Build Status](https://travis-ci.org/mgormley/optimize.svg?branch=master)](https://travis-ci.org/mgormley/optimize)

## Summary

Optimize is a Java library for numerical optimization. Currently the public version includes the following algorithms:

* Stochastic Gradient Descent (SGD) with lots of the tricks from (Bottou,
2012; "Stochastic Gradient Tricks")
* SGD with forward-backward splitting (Duchi & Singer, 2009)
* Truncated gradient (Langford et al., 2008)
* AdaGrad with L1 or L2 regularization 
(Duchi et al., 2011)
* L-BFGS ported from Fortran to C to Java

This release includes contributions from Matt Gormley, Nick Andrews,
Frank Ferraro, and Travis Wolfe.

## Using the Library

The latest public version is deployed on 
[Maven Central](http://search.maven.org/#search%7Cgav%7C1%7Cg%3A%22edu.jhu.hlt.optimize%22%20AND%20a%3A%22optimize%22):

```xml
<dependency>
    <groupId>edu.jhu.hlt.optimize</groupId>
    <artifactId>optimize</artifactId>
    <version>3.1.5</version>
</dependency>
```

## Development

### Build

* Compile the code from the command line:

        mvn compile

* To build a single jar with all the dependencies included:

        mvn compile assembly:single


### Eclipse setup

* Create local versions of the .project and .classpath files for Eclipse:

        mvn eclipse:eclipse

* Add M2\_REPO environment variable to
  Eclipse. http://maven.apache.org/guides/mini/guide-ide-eclipse.html
  Open the Preferences and navigate to 'Java --> Build Path -->
  Classpath Variables'. Add a new classpath variable M2\_REPO with the
  path to your local repository (e.g. ~/.m2/repository).

* To make the project Git aware, right click on the project and select Team -> Git... 
