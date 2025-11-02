# ExogenousAI Tests

This directory contains unit tests for the ExogenousAI project.

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test file:
```bash
python -m pytest tests/test_pipeline.py
```

Run with coverage:
```bash
python -m pytest --cov=src tests/
```

## Test Structure

- `test_pipeline.py`: Core pipeline functionality tests
- Test modules mirror the src/ directory structure

## Writing Tests

When adding new features, please add corresponding tests:

```python
def test_new_feature(self):
    """Test description"""
    # Test implementation
    self.assertEqual(expected, actual)
```

Follow naming convention: `test_<feature>_<aspect>`
