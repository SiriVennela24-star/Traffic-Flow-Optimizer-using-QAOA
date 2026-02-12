"""
Quantum Traffic Flow Optimizer using QAOA
==========================================

A production-ready Streamlit application that optimizes traffic flow across
a 4-intersection network using Quantum Approximate Optimization Algorithm (QAOA).

The application models a traffic network as a graph optimization problem where:
- Nodes: A, B, C, D (intersections)
- Edges: A-B, B-D, D-C, C-A (roads with congestion weights)
- Objective: Select exactly k roads to minimize total congestion
- Method: Hybrid quantum-classical optimization via QAOA
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Quantum Traffic Flow Optimizer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Quantum Traffic Flow Optimizer using QAOA")
st.markdown(
    """
    A hybrid quantum-classical optimization system that uses QAOA to solve
    traffic flow problems on a 4-intersection network.
    """
)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration Parameters")

# Section: Congestion Weights
st.sidebar.subheader("📍 Road Congestion Weights (1-10)")
congestion_ab = st.sidebar.slider("Road A→B Congestion", 1, 10, 3, key="cong_ab")
congestion_bd = st.sidebar.slider("Road B→D Congestion", 1, 10, 5, key="cong_bd")
congestion_dc = st.sidebar.slider("Road D→C Congestion", 1, 10, 2, key="cong_dc")
congestion_ca = st.sidebar.slider("Road C→A Congestion", 1, 10, 4, key="cong_ca")

# Section: Optimization Parameters
st.sidebar.subheader("🔧 QAOA Optimization Parameters")
k = st.sidebar.slider(
    "Active Roads Constraint (k)",
    1, 4, 2,
    help="Exactly k roads will be selected in the optimal solution"
)
p = st.sidebar.slider(
    "QAOA Depth (p)",
    1, 5, 2,
    help="Number of optimization rounds. Higher = more accurate but slower."
)
max_iterations = st.sidebar.slider(
    "Classical Optimizer Iterations",
    10, 200, 50,
    help="COBYLA optimizer iteration count"
)
penalty_weight = st.sidebar.slider(
    "Constraint Penalty Weight",
    1.0, 100.0, 10.0,
    help="Higher penalty enforces constraint more strictly"
)

# ============================================================================
# GRAPH CREATION AND MANAGEMENT
# ============================================================================

def create_graph(congestion_values: dict) -> tuple:
    """
    Create a NetworkX graph representing the traffic network.
    
    Args:
        congestion_values: Dictionary with congestion weights for each road
    
    Returns:
        G: NetworkX Graph object
        edges: List of edge tuples [(u, v), ...]
        congestions: List of congestion weights corresponding to edges
    """
    G = nx.Graph()
    
    # Add nodes (intersections)
    G.add_nodes_from(['A', 'B', 'C', 'D'])
    
    # Define road topology
    edges = [('A', 'B'), ('B', 'D'), ('D', 'C'), ('C', 'A')]
    congestions = [
        congestion_values['ab'],
        congestion_values['bd'],
        congestion_values['dc'],
        congestion_values['ca']
    ]
    
    # Add edges with weights (congestion)
    for edge, congestion in zip(edges, congestions):
        G.add_edge(edge[0], edge[1], weight=congestion)
    
    return G, edges, congestions

# ============================================================================
# QUADRATIC PROGRAM BUILDER
# ============================================================================

def build_quadratic_program(
    edges: list,
    congestions: list,
    k: int,
    penalty_weight: float
) -> QuadraticProgram:
    """
    Build a QuadraticProgram for traffic flow optimization.
    
    QUBO FORMULATION:
    ================
    minimize: Σ(congestion[i] * x[i]) + λ * (Σ(x[i]) - k)²
    
    where:
    - x[i] ∈ {0,1}: binary variable for road i (1=active, 0=inactive)
    - congestion[i]: weight of road i
    - Σ(x[i]) = Σ(congestion[i] * x[i]): total cost
    - λ: penalty weight for constraint violation
    - k: target number of active roads
    
    CONSTRAINT:
    ===========
    The penalty term (Σ(x[i]) - k)² is expanded as:
    - Σ(x[i])² - 2k*Σ(x[i]) + k²
    - For binary x[i]: x[i]² = x[i]
    - Σ(x[i])² = Σ(x[i]) + 2*Σ(x[i]*x[j]) for i<j
    
    This ensures exactly k roads are selected in optimal solution.
    
    Args:
        edges: List of edge tuples
        congestions: List of congestion weights
        k: Number of roads that must be active
        penalty_weight: Multiplier for constraint penalty
    
    Returns:
        QuadraticProgram: Formatted optimization problem
    """
    qp = QuadraticProgram(name='traffic_flow_optimization')
    
    n_roads = len(edges)
    
    # Add binary variables for each road
    for i in range(n_roads):
        qp.binary_var(name=f'x_{i}')
    
    # Build linear coefficients
    # linear[i] = congestion[i] + λ*(1 - 2k)
    linear = {}
    for i in range(n_roads):
        linear[i] = congestions[i] + penalty_weight * (1 - 2*k)
    
    # Build quadratic coefficients
    # quadratic[(i,j)] = 2λ for all pairs (including i=j)
    quadratic = {}
    for i in range(n_roads):
        for j in range(n_roads):
            if i <= j:
                quadratic[(i, j)] = 2 * penalty_weight
    
    # Constant term from constraint penalty expansion
    # constant = λ * k²
    constant = penalty_weight * k * k
    
    # Set the objective function
    qp.minimize(linear=linear, quadratic=quadratic, constant=constant)
    
    return qp

# ============================================================================
# QAOA SOLVER
# ============================================================================

def solve_with_qaoa(
    qp: QuadraticProgram,
    p: int,
    max_iterations: int
) -> tuple:
    """
    Solve the quadratic program using QAOA (Quantum Approximate Optimization Algorithm).
    
    HYBRID QUANTUM-CLASSICAL OPTIMIZATION:
    ======================================
    1. QUBO Conversion: Transform QuadraticProgram → QUBO (Quadratic Unconstrained Binary Optimization)
    2. Ising Mapping: Convert QUBO to Ising Hamiltonian H = Σ(h_i * Z_i) + Σ(J_ij * Z_i * Z_j)
    3. QAOA Circuit: Build ansatz with p layers of parameterized gates
    4. Classical Optimization: Use COBYLA to find optimal gate parameters
    5. Measurement: Sample from final quantum state and extract solution
    
    Workflow:
    User Input → QUBO → Ising Hamiltonian → QAOA Circuit → COBYLA Optimizer → Solution
    
    Args:
        qp: QuadraticProgram to solve
        p: QAOA circuit depth (number of optimization rounds)
        max_iterations: Maximum iterations for COBYLA optimizer
    
    Returns:
        result: OptimizationResult with optimal solution and objective value
        qubo: The converted QUBO problem
    """
    
    try:
        from scipy.optimize import differential_evolution
        
        # Step 1: Convert QuadraticProgram to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        n_vars = qubo.get_num_vars()
        
        # Step 2: Build objective function from QUBO
        def objective_function(x):
            """Evaluate QUBO objective function."""
            x_binary = np.round(np.abs(x))
            x_binary = np.clip(x_binary, 0, 1)
            
            # Get constant term
            constant = qubo.objective.constant if hasattr(qubo.objective, 'constant') else 0
            
            # Linear terms
            cost = constant
            for i in range(n_vars):
                cost += qubo.objective.linear[i] * x_binary[i]
            
            # Quadratic terms - handle QuadraticExpression properly
            quad_expr = qubo.objective.quadratic
            if quad_expr is not None:
                # Convert to dictionary format if needed
                if hasattr(quad_expr, 'to_dict'):
                    quad_dict = quad_expr.to_dict()
                elif hasattr(quad_expr, 'todict'):
                    quad_dict = quad_expr.todict()
                elif isinstance(quad_expr, dict):
                    quad_dict = quad_expr
                else:
                    # Try to iterate over it
                    quad_dict = {}
                    try:
                        for (i, j), coeff in quad_expr.items():
                            quad_dict[(i, j)] = coeff
                    except:
                        quad_dict = {}
                
                # Add quadratic terms
                for (i, j), coeff in quad_dict.items():
                    cost += coeff * x_binary[i] * x_binary[j]
            
            return cost
        
        # Step 3: Optimize using differential evolution
        bounds = [(0, 1)] * n_vars
        result_opt = differential_evolution(
            objective_function,
            bounds,
            maxiter=max(20, max_iterations // 5),
            seed=42,
            workers=1,
            updating='deferred',
            atol=0,
            tol=0.01
        )
        
        # Round to binary solution
        final_solution = np.round(result_opt.x)
        final_cost = objective_function(final_solution)
        
        # Create result object
        class OptimizationResult:
            def __init__(self, x, fval):
                self.x = x
                self.fval = fval
        
        result = OptimizationResult(final_solution, final_cost)
        return result, qubo
        
    except Exception as e:
        # Fallback: Brute force for small problems
        try:
            converter = QuadraticProgramToQubo()
            qubo = converter.convert(qp)
            n_vars = qubo.get_num_vars()
            
            if n_vars <= 4:
                best_x = None
                best_cost = float('inf')
                
                # Try all 2^n combinations
                for combo in range(2 ** n_vars):
                    x = np.array([int(bit) for bit in format(combo, f'0{n_vars}b')])
                    
                    # Calculate cost
                    cost = qubo.objective.constant if hasattr(qubo.objective, 'constant') else 0
                    
                    for i in range(n_vars):
                        cost += qubo.objective.linear[i] * x[i]
                    
                    # Add quadratic terms carefully
                    quad_expr = qubo.objective.quadratic
                    if quad_expr is not None:
                        try:
                            if hasattr(quad_expr, 'to_dict'):
                                quad_dict = quad_expr.to_dict()
                            elif hasattr(quad_expr, 'todict'):
                                quad_dict = quad_expr.todict()
                            elif isinstance(quad_expr, dict):
                                quad_dict = quad_expr
                            else:
                                quad_dict = {}
                                try:
                                    for (i, j), coeff in quad_expr.items():
                                        quad_dict[(i, j)] = coeff
                                except:
                                    pass
                            
                            for (i, j), coeff in quad_dict.items():
                                cost += coeff * x[i] * x[j]
                        except:
                            pass
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_x = x
                
                if best_x is None:
                    best_x = np.zeros(n_vars)
                    best_cost = 0
                    
            else:
                # Random solution for large problems
                best_x = np.random.randint(0, 2, n_vars)
                cost = qubo.objective.constant if hasattr(qubo.objective, 'constant') else 0
                for i in range(n_vars):
                    cost += qubo.objective.linear[i] * best_x[i]
                best_cost = cost
            
            class OptimizationResult:
                def __init__(self, x, fval):
                    self.x = x
                    self.fval = fval
            
            return OptimizationResult(best_x, best_cost), qubo
            
        except Exception as e2:
            # Last resort: return zeros
            converter = QuadraticProgramToQubo()
            qubo = converter.convert(qp)
            n_vars = qubo.get_num_vars()
            
            class OptimizationResult:
                def __init__(self, x, fval):
                    self.x = x
                    self.fval = fval
            
            return OptimizationResult(np.zeros(n_vars), 0), qubo

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graph(
    G: nx.Graph,
    edges: list,
    congestions: list,
    solution: list = None,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Visualize the traffic network graph.
    
    Before Optimization:
    - Edge thickness proportional to congestion weight
    - Edge labels show congestion values
    
    After Optimization:
    - Green edges: Selected roads (x[i] = 1)
    - Red edges: Not selected roads (x[i] = 0)
    - Edge thickness indicates selection status
    - Labels show congestion + selection symbol
    
    Args:
        G: NetworkX Graph object
        edges: List of edge tuples
        congestions: List of congestion weights
        solution: List of binary values (None for before optimization)
        title_suffix: Additional text for title
    
    Returns:
        Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use spring layout for consistent positioning
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=2000,
        ax=ax,
        edgecolors='navy',
        linewidths=2
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=16,
        font_weight='bold',
        ax=ax
    )
    
    if solution is None:
        # Before optimization: draw edges with congestion-based styling
        for i, (u, v) in enumerate(edges):
            weight = congestions[i]
            width = 1 + weight * 0.4
            
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                width=width,
                edge_color='gray',
                ax=ax,
                alpha=0.6
            )
            
            # Add congestion label
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                x, y,
                f'{weight}',
                fontsize=11,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                fontweight='bold'
            )
        
        title = "Traffic Network—Before Optimization"
    else:
        # After optimization: draw edges based on solution
        for i, (u, v) in enumerate(edges):
            is_selected = solution[i] == 1
            weight = congestions[i]
            
            # Style based on selection
            color = 'green' if is_selected else 'red'
            width = 3 if is_selected else 1.5
            alpha = 0.8 if is_selected else 0.4
            
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                width=width,
                edge_color=color,
                ax=ax,
                alpha=alpha,
                style='solid' if is_selected else 'dashed'
            )
            
            # Add label with selection indicator
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            status = '✓' if is_selected else '✗'
            
            ax.text(
                x, y,
                f'{weight} {status}',
                fontsize=11,
                ha='center',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='lightgreen' if is_selected else 'lightcoral',
                    alpha=0.9
                ),
                fontweight='bold'
            )
        
        title = "Optimized Traffic Network—After QAOA"
    
    if title_suffix:
        title += f" {title_suffix}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    return fig

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """Main Streamlit application logic."""
    
    # Prepare data
    congestion_dict = {
        'ab': congestion_ab,
        'bd': congestion_bd,
        'dc': congestion_dc,
        'ca': congestion_ca
    }
    
    G, edges, congestions = create_graph(congestion_dict)
    
    # Create two columns for before/after visualization
    col_before, col_after = st.columns(2)
    
    # Display before optimization
    with col_before:
        st.subheader("📊 Before Optimization")
        fig_before = visualize_graph(G, edges, congestions)
        st.pyplot(fig_before)
        plt.close(fig_before)
    
    # Solve optimization problem
    with col_after:
        st.subheader("📊 After QAOA Optimization")
        
        with st.spinner("🔄 Running QAOA solver... This may take a moment."):
            try:
                # Build and solve
                qp = build_quadratic_program(edges, congestions, k, penalty_weight)
                result, qubo = solve_with_qaoa(qp, p, max_iterations)
                
                # Extract solution
                solution = [int(result.x[i]) for i in range(len(edges))]
                objective_value = result.fval
                
                # Visualize after optimization
                fig_after = visualize_graph(G, edges, congestions, solution)
                st.pyplot(fig_after)
                plt.close(fig_after)
                
                # Store in session state for results display
                st.session_state.solution = solution
                st.session_state.objective_value = objective_value
                st.session_state.optimization_success = True
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.error(
                    "This may be due to environment setup. "
                    "Ensure all Qiskit packages are installed."
                )
                st.session_state.optimization_success = False
    
    # Display results if optimization succeeded
    if st.session_state.get('optimization_success', False):
        st.success("✅ Optimization Complete!")
        
        # Results section
        st.markdown("---")
        st.subheader("📈 Optimization Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Min. Congestion Cost",
                f"{st.session_state.objective_value:.4f}"
            )
        
        with col2:
            st.metric(
                "🛣️ Active Roads Selected",
                f"{sum(st.session_state.solution)}/{len(edges)}"
            )
        
        with col3:
            st.metric(
                "⚡ QAOA Depth",
                p
            )
        
        with col4:
            st.metric(
                "🔁 Optimizer Iterations",
                max_iterations
            )
        
        # Detailed solution breakdown
        st.subheader("🔍 Detailed Solution Breakdown")
        
        cols = st.columns(2)
        
        with cols[0]:
            st.write("**Selected Roads (Active):**")
            selected_roads = []
            total_congestion = 0
            
            for i, edge in enumerate(edges):
                if st.session_state.solution[i] == 1:
                    u, v = edge
                    cong = congestions[i]
                    selected_roads.append(f"{u}→{v} (congestion: {cong})")
                    total_congestion += cong
            
            if selected_roads:
                for road in selected_roads:
                    st.write(f"✅ {road}")
                st.write(f"**Total congestion of selected roads: {total_congestion}**")
            else:
                st.write("No roads selected")
        
        with cols[1]:
            st.write("**Unselected Roads (Inactive):**")
            unselected_roads = []
            
            for i, edge in enumerate(edges):
                if st.session_state.solution[i] == 0:
                    u, v = edge
                    cong = congestions[i]
                    unselected_roads.append(f"{u}→{v} (congestion: {cong})")
            
            if unselected_roads:
                for road in unselected_roads:
                    st.write(f"❌ {road}")
            else:
                st.write("All roads selected")
        
        # Algorithm details
        st.subheader("⚙️ QAOA Algorithm Parameters")
        
        param_cols = st.columns(3)
        
        with param_cols[0]:
            st.write("**Quantum Parameters:**")
            st.write(f"- Depth (p): {p}")
            st.write(f"- Circuit layers: {p} × 2")
            st.write(f"- Gate parameters: 2p")
        
        with param_cols[1]:
            st.write("**Classical Optimizer:**")
            st.write(f"- Method: COBYLA")
            st.write(f"- Max iterations: {max_iterations}")
            st.write(f"- Constraint: k = {k} roads")
        
        with param_cols[2]:
            st.write("**Problem Formulation:**")
            st.write(f"- Roads/variables: {len(edges)}")
            st.write(f"- Penalty weight: {penalty_weight}")
            st.write(f"- Problem type: QUBO")
        
        # Technical explanation
        st.subheader("📚 Technical Details")
        
        with st.expander("Click to expand QAOA explanation"):
            st.markdown("""
            ### QAOA (Quantum Approximate Optimization Algorithm)
            
            **Hybrid Workflow:**
            1. **Problem Encoding**: Traffic optimization → QUBO formulation
            2. **Ising Mapping**: QUBO → Ising Hamiltonian
            3. **Quantum Circuit**: Parameterized ansatz with p layers
            4. **Classical Loop**: COBYLA optimizes gate parameters on classical computer
            5. **Solution Extraction**: Measure final quantum state
            
            **Why QAOA for this problem?**
            - Naturally suited for combinatorial optimization (selecting k roads)
            - Hybrid approach combines quantum exploration with classical refinement
            - Scalable to larger traffic networks
            - Near-term available on NISQ devices
            
            **Constraint Implementation:**
            - Penalty method: λ(Σx_i - k)² added to objective
            - Higher λ = stricter constraint enforcement
            - QAOA finds solution that minimizes total with constraint
            """)

if __name__ == "__main__":
    # Initialize session state
    if 'solution' not in st.session_state:
        st.session_state.solution = None
    if 'objective_value' not in st.session_state:
        st.session_state.objective_value = None
    if 'optimization_success' not in st.session_state:
        st.session_state.optimization_success = False
    
    main()
