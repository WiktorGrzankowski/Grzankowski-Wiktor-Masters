### Project description
This project is part of a Master's Thesis of Wiktor Grzankowski at the University of Warsaw under the supervision of professor Piotr Skowron.

The thesis "Stable Priceability for Additive Utilities" extends the classic notion of stable priceability axiom, first defined in the paper [Market-Based Explanations of Collective Decisions](https://www.cs.toronto.edu/~nisarg/papers/priceability.pdf)
to the setting of elections with cardinal ballots cast.

This project contains scripts and outputs of those scripts, which are used in chapter 4 of the thesis to evaluate proposed axioms on real-life data.

Another significant part of the code is merged into the [Pabutools library repository on GitHub](https://github.com/COMSOC-Community/pabutools).

### Scripts
There are 3 main scripts:
1. analysis_main.py -> reads all instances from resources and evaluates them, in ascending order, using 5 voting rules. Output of each
rule is checked if it is priceable or stable priceable.
2. any_priceable_main.py -> reads all instances with their profile types and checks if any method returned a priceable/stable priceable under
exhaustive/non-exhaustive conditions. For those, where no satisfying solution was found, tries to find the solution - calls priceable(...) method with no initial budget allocation.
3. plots_main.py -> creates plots for convenient analysis, based on outputs from previous scripts.

### Thesis
Latex code and the full PDF of the thesis is also attached for the convenience of future researchers.
