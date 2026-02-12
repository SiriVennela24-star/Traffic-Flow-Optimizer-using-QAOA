# Quick Start Guide (2 Minutes)

## 30-Second Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python -m streamlit run app.py
   ```

3. **Open browser:**
   Navigate to `http://localhost:8501`

## First Run (2 Minutes)

### Step 1: Default Configuration
The app loads with default settings:
- Road congestions: A-B=3, B-D=5, D-C=2, C-A=4
- Active roads to select: k=2
- QAOA depth: p=2
- Optimizer iterations: 50

Adjust the sliders on the left sidebar to modify these values.

### Step 2: Observe the Visualizations
- **Left Graph**: Shows original network with congestion values
- **Right Graph**: Shows optimized network after running QAOA
  - Yellow edges = selected roads ✓
  - Black edges = unselected roads ✗

### Step 3: Interpret Results
- **Min. Congestion Cost**: Total cost of selected roads
- **Active Roads Selected**: How many roads were chosen
- **QAOA Depth**: Quantum circuit depth used
- **Optimizer Iterations**: Classical optimization steps

## Common Use Cases

### Case 1: Find Cheapest 2-Road Route
```
Congestions: A-B=2, B-D=8, D-C=3, C-A=7
k = 2
p = 2
→ Solution: Select A-B + D-C (total cost: 5)
→ Roads B-D and C-A are avoided (too expensive)
```

### Case 2: Compare QAOA Depths
```
Run with p=1 (fast):     Note the solution and cost
Run with p=3 (slower):   Compare quality vs computation time
→ Higher p generally gives better but slower results
```

### Case 3: Understand Constraint Penalty
```
Low penalty (5):   Solution might violate k constraint
Medium penalty (10): Good balance
High penalty (50):  Strictly enforces k, might compromise cost
```

## Tips & Tricks

✨ **Quick Testing:**
- p=1, iterations=25 for fast feedback
- p=2, iterations=50 for balanced results
- p=3, iterations=100 for high accuracy

✨ **Experiment Ideas:**
- Set all congestions equal → See which k roads are selected
- Set one road very expensive → Verify it's not selected (with high k)
- Increase penalty weight → Ensure exactly k roads selected
- Reduce penalty weight → See if k constraint is violated

✨ **Key Parameters:**

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| k | 1-4 | 2 | Roads to select |
| p | 1-5 | 2 | QAOA depth |
| Iterations | 10-200 | 50 | Optimization steps |
| Penalty | 1-100 | 10 | Constraint strength |

## Troubleshooting

**Q: App running slow?**
- A: Reduce p to 1, reduce iterations to 25

**Q: Results don't satisfy k constraint?**
- A: Increase penalty_weight to 20-50

**Q: Want to understand the algorithm?**
- A: Click "Technical Details" expander for explanation

**Q: How to run on network?**
- A: Replace localhost with your machine IP address

## Next Steps

1. ✅ Run the app with default settings
2. 📊 Experiment with 5-10 different parameter combinations
3. 📖 Read README.md for mathematical details
4. 🔬 Modify congestion values to model real scenarios
5. 🚀 Deploy to Streamlit Cloud for sharing

---

**Ready to optimize! 🚀 Run the app now!**
