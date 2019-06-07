A sample Python project
=======================

This package is to investigate a possible security issue with pip changing
the --user syntax for Travis projects.

https://pip.pypa.io/en/stable/user_guide/#user-installs

pip install --user $USER git+https://github.com/google/closure-linter.git

results in

```
Could not find any downloads that satisfy the requirement travis
```

See e.g. https://github.com/travis-ci/travis-ci/issues/6905


