# 🚦 Quantum Traffic Flow Optimizer using QAOA

A production-ready Streamlit web application implementing a hybrid quantum-classical optimization workflow for traffic flow problems using Qiskit's Quantum Approximate Optimization Algorithm (QAOA).

## Overview

This application solves a realistic traffic network optimization problem where the goal is to select exactly **k** roads from a network of 4 intersections to minimize total congestion. It demonstrates modern quantum computing techniques applied to classical optimization problems.

### Problem Statement

**Network Topology:**
- 4 Intersections: A, B, C, D
- 4 Roads: A→B, B→D, D→C, C→A (forming a cycle)
- Each road has a congestion weight (1-10)
- Objective: Select exactly k roads to minimize total congestion

**Mathematical Formulation:**

```
minimize: Σ(congestion[i] × x[i]) + λ × (Σ(x[i]) - k)²

subject to:
  x[i] ∈ {0, 1}  ∀ i
  Σ(x[i]) = k    (exactly k roads active)
```

Where:
- `x[i]`: Binary variable (1=road active, 0=road inactive)
- `congestion[i]`: Weight of road i
- `k`: Constraint on number of active roads
- `λ`: Penalty weight for constraint violation

## Installation & Quick Start

### Requirements
- Python 3.9+
- pip package manager

### Installation Steps

```bash
# Navigate to project folder
cd "C:\Quantum Traffic Optimizer"

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Features

✅ **Interactive Streamlit UI**
- Real-time slider controls for all parameters
- Live visualization of network before/after optimization
- Responsive to parameter changes

✅ **Complete QAOA Implementation**
- QuadraticProgram formulation
- QUBO conversion with penalty methods
- QAOA with configurable depth
- COBYLA classical optimizer

✅ **Rich Visualizations**
- Graph rendering with NetworkX layout
- Before optimization: edges sized by congestion
- After optimization: yellow (selected) vs black (unselected) edges
- Congestion weight labels on edges

✅ **Detailed Results Display**
- Optimal solution vector
- Minimum congestion cost
- Selected vs unselected roads
- Algorithm parameters summary
- Technical explanation section

✅ **Enhanced UI Design**
- 🟦 Blue: Primary color for quantum/classical components
- 🟨 Yellow: Highlights optimized/selected elements
- ⬛ Black: Represents unselected/inactive elements
- Professional gradient backgrounds
- Color-coded result sections

## Usage Guide

### 1. Configure Road Congestion (Sidebar)

Use the sliders to set congestion weights (1-10) for each road:
- **Road A→B**: Cost of traveling from intersection A to B
- **Road B→D**: Cost of traveling from intersection B to D  
- **Road D→C**: Cost of traveling from intersection D to C
- **Road C→A**: Cost of traveling from intersection C to A

### 2. Configure QAOA Parameters (Sidebar)

**Active Roads Constraint (k)**
- Number of roads that must be selected (1-4)
- Example: k=2 means exactly 2 roads will be in the solution

**QAOA Depth (p)**
- Number of optimization layers (1-5)
- p=1: Fast, shallow circuit, lower accuracy
- p=3-5: Better quality solutions, more computation
- Recommended: p=2-3 for balance

**Classical Optimizer Iterations**
- COBYLA refinement steps (10-200)
- Higher values → better parameter optimization
- Recommended: 50-100

**Constraint Penalty Weight**
- Multiplier for constraint enforcement (1-100)
- Higher penalty → stricter constraint satisfaction
- Recommended: 10-20

### 3. Interpret Results

**Before/After Visualizations**
- Left panel: Original network with congestion values
- Right panel: Optimized network with selections marked

**Metrics Dashboard**
- Minimum Congestion Cost: Total cost of selected roads
- Active Roads Selected: Number of selected roads (should equal k)
- QAOA Depth: Configuration used
- Optimizer Iterations: Configuration used

**Solution Breakdown**
- Selected roads marked with 🟨
- Unselected roads marked with ⬛
- Total congestion calculated

## Qiskit Components Used

| Component | Purpose |
|-----------|---------|
| **QuadraticProgram** | Problem formulation with binary variables and constraints |
| **QuadraticProgramToQubo** | Conversion to Quadratic Unconstrained Binary Optimization |
| **QAOA** | Quantum Approximate Optimization Algorithm implementation |
| **COBYLA** | Constrained Optimization By Linear Approximation (classical) |
| **MinimumEigenOptimizer** | Wrapper for hybrid quantum-classical optimization |
| **Sampler** | Quantum circuit execution and measurement |
| **Aer Simulator** | Cloud-based quantum circuit simulator |

## Architecture

### Hybrid Quantum-Classical Workflow

```
User Input (Streamlit UI)
        ↓
Graph Modeling (NetworkX)
        ↓
QUBO Formulation (QuadraticProgram)
        ↓
QUBO Conversion
        ↓
Ising Hamiltonian Mapping
        ↓
QAOA Circuit Construction
        ↓
Classical Optimization (COBYLA)
        ↓
Quantum Circuit Execution
        ↓
Measurement & Solution Extraction
        ↓
Visualization & Results Display
```

## File Guide

```
C:\Quantum Traffic Optimizer\
├── app.py              ← Main Streamlit application (RUN THIS)
├── requirements.txt    ← Python package dependencies
├── README.md           ← Full documentation (this file)
├── QUICKSTART.md       ← 30-second setup & examples
├── CONFIGURATION.md    ← Parameter tuning guide
├── INSTALL.md          ← Installation & deployment
└── .gitignore          ← Git configuration
```

## Example Scenarios

### Scenario 1: Find Cheapest 2-Road Route
```
Congestions: A-B=2, B-D=8, D-C=3, C-A=7
k = 2
p = 2
→ Solution: Select A-B (2) + D-C (3) = Cost 5
```

### Scenario 2: Understand QAOA Quality
```
Set: p=1, iterations=10  (fast, lower quality)
Then: p=3, iterations=100 (slow, better quality)
Compare the results on same data
```

### Scenario 3: Constraint Penalty Tuning
```
High penalty (50): Strict k constraint, may find sub-optimal solution
Low penalty (5): Faster convergence, might violate constraint
Sweet spot: 10-20
```

## Troubleshooting

**Q: Why is the app slow?**
A: Reduce p to 1 and iterations to 25. Start simple!

**Q: Does solution always have k roads?**
A: Should be yes if penalty_weight is sufficient. Increase it if not.

**Q: Can I use this for larger networks?**
A: Current app handles 4 nodes. Larger requires code modification.

**Q: What if results seem random?**
A: This is quantum! Run multiple times with same settings. Average converges.

## References

1. **QAOA Paper**: Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (2014)
2. **Qiskit Documentation**: https://qiskit.org/documentation/
3. **Optimization in Qiskit**: https://qiskit.org/documentation/stubs/qiskit_optimization.html
4. **QUBO Problems**: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

## License

MIT License - Feel free to use for education and research.

---

**Built with ❤️ using Streamlit, Qiskit, and NetworkX**

For questions or issues, please refer to the documentation files or modify the code as needed.
