# Residuum KENN

Residuum KENN (Knowledge Enhanced Neural Networks) is an extended version of KENN2 (https://github.com/HEmile/KENN-PyTorch) that allows to define implication rules.

**NB:** version 1.0 of KENN was released for Python 2.7 and TensorFlow 1.x and it is available at [KENN v1.0](https://github.com/DanieleAlessandro/KENN). Notice that this version is not backward compatible. Additionally, this implementation of KENN can work with relational domains, meaning that one can use also binary predicates to express logical rules which involve the relationship between two objects. There is also an implementation for Tensorflow 2 available at [KENN v2](https://github.com/DanieleAlessandro/KENN2).

The implementation of KENN2 model is presented in paper:
[Knowledge Enhanced Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-29908-8_43).



### **2. Add implication rules**

There are two rows: the first containing the list of unary predicates, the second containing the binary predicates.
The clauses are also split in four groups: the first group contains only clauses with unary predicates, the second clauses with both unary and binary predicates, the third contains implication rules with only unary predicates, and the fourth clauses with both unary and binary prediactes. The groups are separated by a row containing the `>` symbol.

Unary predicates are defined on a single variable (e.g. _Dog(x)_), binary predicates on two variables separated by a dot (e.g. _Friends(x.y)_).
Each clause is in a separate row and must be written respecting this properties:

- Logical disjunctions are represented with commas;
- Logical conjunction is represented with semicolon;
- Implication is represented with ->
- If a literal is negated, it must be preceded by the lowercase 'n';
- They must contain only predicates specified in the first row;
- There shouldn't be spaces.


```
Smoker,Cancer
Friends

_:nSmoker(x),Cancer(x)
>
>
>
_:Friends(x.y);Smoker(x)->Smoker(y)
```

The first row specifies that there are two unary predicates: Smoker and Cancer. Second row specifies the binary predicates, which in this case is one: Friends. The first clause encodes the fact that a smoker also has cancer (note that the rules does not represent hard constraints) and the fourth, which contains also the binary predicate, expresses the idea that friends tend to have similar smoking habits.

### **3. Residuum semantics**

Implications rules are intepreted by the model using the fuzzy logic Residuum semantics. This means that truth value of implication rules are computed differently by the model. And so, also the modifications to perform on the predicates truth values.

To have an overview on the function of KENN refer to [Knowledge Enhanced Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-29908-8_43).

## License

Copyright (c) 2021, Daniele Alessandro, Mazzieri Riccardo, Serafini Luciano, van Krieken Emile, Andrea Mascari
All rights reserved.

Licensed under the BSD 3-Clause License.
