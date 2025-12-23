# Getting Started

## Installation

Install relperm using uv:

```bash
uv pip install relperm
```

## Basic Usage

Here's a simple example:

```python
import numpy as np
from relperm import calculate_saturation

# Define saturation values
sw = np.linspace(0, 1, 100)

# Calculate relative permeability
kr = calculate_saturation(sw)
```
