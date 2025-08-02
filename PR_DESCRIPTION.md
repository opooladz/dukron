# Add PSGD Quad

Adds PSGD Quad optimizer to dukron.

## Changes
- Add `dukron/_quad.py`
- Update exports in `__init__.py`
- Add test in `tests/test_quad.py`
- Update README

## Usage
```python
from dukron import quad

optimizer = quad(learning_rate=0.01)
```

Based on levanter implementation by @evanatyourservice.