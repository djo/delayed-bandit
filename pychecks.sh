#!/bin/bash

black_code=0
flake_code=0
mypy_code=0
pytest_code=0

python -m black -S --line-length 100 ${justdoit:+--check} delayed_bandit
black_code=$?

python -m flake8  --exclude snippets.py --max-line-length 100 delayed_bandit
flake_code=$?

python -m mypy delayed_bandit
mypy_code=$?

python -m pytest
pytest_code=$?

exit_code=0

for code in "black_code" "flake_code" "mypy_code" "pytest_code"; do
  if [ "${!code}" != 0 ]
  then
    echo "$code FAILED";
    exit_code=1;
  else
    echo "$code OK";
  fi
done

exit $exit_code;
