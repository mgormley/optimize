

# Summary

This is the pre-release of the Hopkins Optimization
Library. Currently, the public version includes an implementation of
Stochastic Gradient Descent with lots of the tricks from (Bottou,
2012; "Stochastic Gradient Tricks").

This release includes contributions from Matt Gormley, Nick Andrews,
Frank Ferraro, and Travis Wolfe.

# Using the Library

The latest version is deployed on Maven Central:

    <dependency>
      <groupId>edu.jhu.hlt.optimize</groupId>
      <artifactId>optimize</artifactId>
      <version>2.0.1</version>
    </dependency>

## Build

* Compile the code from the command line:

        mvn compile

* To build a single jar with all the dependencies included:

        mvn compile assembly:single

# Development

## Eclipse setup

* Create local versions of the .project and .classpath files for Eclipse:

        mvn eclipse:eclipse

* Add M2_REPO environment variable to
  Eclipse. http://maven.apache.org/guides/mini/guide-ide-eclipse.html
  Open the Preferences and navigate to 'Java --> Build Path -->
  Classpath Variables'. Add a new classpath variable M2_REPO with the
  path to your local repository (e.g. ~/.m2/repository).

* To make the project Git aware, right click on the project and select Team -> Git... 
