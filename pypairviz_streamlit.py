import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from dataclasses import dataclass
import math

"""
pyPairViz-Streamlit: A web-based implementation of molecular pair potential visualization
Based on the original pyPairViz (https://github.com/njcorrente/pyPairViz) by Nick Corrente
This version adapts the functionality to a web interface using Streamlit
"""

# Configure the Streamlit page
st.set_page_config(
    page_title="Molecular Interaction Potential Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# Molecular Interaction Potential Visualizer\n\nBased on the original [pyPairViz](https://github.com/njcorrente/pyPairViz) by Nick Corrente."
    }
)

# Model definitions
@dataclass
class PotentialModel:
    epsilon_over_kB: float = 120.0  # K
    sigma: float = 3.4  # Angstrom
    name: str = "Base Model"
    description: str = "Base model description"
    equation: str = "Base equation"

    def calculate(self, r):
        return np.zeros_like(r)

class LennardJones(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Lennard-Jones"
        self.description = "Most commonly used for noble gases and simple molecules. Combines short-range repulsion (r⁻¹²) with longer-range attraction (r⁻⁶)."
        self.equation = r"V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]"

    def calculate(self, r):
        return 4 * self.epsilon_over_kB * ((self.sigma/r)**12 - (self.sigma/r)**6)

class MorsePotential(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Morse"
        self.a = 1.0  # width parameter
        self.description = "Common for diatomic molecules. Provides more realistic behavior for molecular vibrations than Lennard-Jones."
        self.equation = r"V(r) = Dₑ[1 - e^{-a(r-rₑ)}]²"

    def calculate(self, r):
        return self.epsilon_over_kB * (1 - np.exp(-self.a * (r - self.sigma)))**2

class BuckinghamPotential(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Buckingham"
        self.A = 1000.0  # repulsive strength
        self.B = 2.0     # repulsive range
        self.description = "Alternative to Lennard-Jones with exponential repulsion. Often more accurate at short ranges."
        self.equation = r"V(r) = Ae^{-Br} - C/r⁶"

    def calculate(self, r):
        return self.A * np.exp(-self.B * r) - self.epsilon_over_kB * (self.sigma/r)**6

class YukawaPotential(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Yukawa"
        self.kappa = 1.0  # screening length
        self.description = "Used in plasma physics and colloidal systems. Represents screened electrostatic interactions."
        self.equation = r"V(r) = (ε/r)e^{-κr}"

    def calculate(self, r):
        return (self.epsilon_over_kB/r) * np.exp(-self.kappa * r)

class MiePotential(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Mie"
        self.n = 12  # repulsive exponent
        self.m = 6   # attractive exponent
        self.description = "Generalized form of Lennard-Jones with adjustable exponents."
        self.equation = r"V(r) = ε[(σ/r)ⁿ - (σ/r)ᵐ]"

    def calculate(self, r):
        return self.epsilon_over_kB * ((self.sigma/r)**self.n - (self.sigma/r)**self.m)

class HardSphere(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Hard Sphere"
        self.description = "Simplest model where particles act as perfect rigid spheres."
        self.equation = "V(r) = ∞ for r < σ, 0 for r ≥ σ"

    def calculate(self, r):
        mask = r < self.sigma
        potential = np.zeros_like(r)
        potential[mask] = np.inf
        return potential

class SquareWell(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Square Well"
        self.well_width = 1.5
        self.well_depth = 1.0
        self.description = "Combines hard sphere repulsion with a constant attractive well."
        self.equation = "V(r) = ∞ for r < σ, -ε for σ ≤ r < λσ, 0 for r ≥ λσ"

    def calculate(self, r):
        well_position = self.sigma * self.well_width
        mask_core = r < self.sigma
        mask_well = (r >= self.sigma) & (r < well_position)
        
        potential = np.zeros_like(r)
        potential[mask_core] = np.inf
        potential[mask_well] = -self.epsilon_over_kB * self.well_depth
        return potential

class Sutherland(PotentialModel):
    def __init__(self, epsilon_over_kB=120.0, sigma=3.4):
        super().__init__(epsilon_over_kB, sigma)
        self.name = "Sutherland"
        self.n = 12
        self.description = "Historical potential with hard-core repulsion and power-law attraction."
        self.equation = r"V(r) = ∞ for r < σ, -ε(σ/r)ⁿ for r ≥ σ"

    def calculate(self, r):
        if isinstance(r, np.ndarray):
            mask = r < self.sigma
            potential = -self.epsilon_over_kB * (self.sigma/r)**self.n
            potential[mask] = np.inf
            return potential
        else:
            if r < self.sigma:
                return np.inf
            else:
                return -self.epsilon_over_kB * (self.sigma/r)**self.n

def create_molecule_visualization(distance, sigma):
    # Calculate positions
    left_x = 0
    right_x = distance
    molecule_radius = sigma/2
    
    # Create figure with equal aspect ratio
    fig = go.Figure()
    
    # Left molecule
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=left_x - molecule_radius, 
        y0=-molecule_radius,
        x1=left_x + molecule_radius, 
        y1=molecule_radius,
        fillcolor="blue",
        line_color="darkblue",
    )
    
    # Right molecule
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=right_x - molecule_radius, 
        y0=-molecule_radius,
        x1=right_x + molecule_radius, 
        y1=molecule_radius,
        fillcolor="red",
        line_color="darkred",
    )
    
    # Add horizontal line through centers
    fig.add_shape(
        type="line",
        x0=left_x, 
        y0=0,
        x1=right_x, 
        y1=0,
        line=dict(color="black", width=1),
    )
    
    # Add dashed distance line
    fig.add_shape(
        type="line",
        x0=left_x, 
        y0=-molecule_radius*1.5,
        x1=right_x, 
        y1=-molecule_radius*1.5,
        line=dict(color="black", dash="dash"),
    )
    
    # Distance label
    fig.add_annotation(
        x=(left_x + right_x)/2,
        y=-molecule_radius*2,
        text=f"r = {distance:.2f} Å",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update layout with fixed aspect ratio
    plot_margin = sigma * 1.5
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            range=[left_x-plot_margin, right_x+plot_margin],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="y",  # This ensures equal scaling
            scaleratio=1      # Force 1:1 aspect ratio
        ),
        yaxis=dict(
            range=[-plot_margin, plot_margin],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        width=600,  # Fixed width
        height=300  # Fixed height
    )
    
    return fig

def create_potential_plot(model, r, V, distance, current_V, y_max_factor=10):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set consistent y-axis limits
    y_max = model.epsilon_over_kB * y_max_factor
    y_min = -2 * model.epsilon_over_kB
    ax.set_ylim([y_min, y_max])
    
    # Add horizontal line at V = 0
    ax.axhline(y=0, color='gray', linewidth=1.5, linestyle='-', alpha=0.5)
    
    # Plot based on model type
    if isinstance(model, (HardSphere, Sutherland)):
        valid_mask = ~np.isinf(V)
        ax.plot(r[valid_mask], V[valid_mask], '-', 
                color='#2E86C1', linewidth=2.5, alpha=0.8)
        
        r_repulsive = r[r < model.sigma]
        if len(r_repulsive) > 0:
            V_repulsive = np.full_like(r_repulsive, y_max)
            ax.plot(r_repulsive, V_repulsive, '--', 
                   color='#e74c3c', linewidth=2, alpha=0.5)
            
            ax.vlines(x=model.sigma, ymin=V[r >= model.sigma][0], ymax=y_max,
                     colors='#e74c3c', linestyles='--', alpha=0.7)
    
    elif isinstance(model, SquareWell):
        well_position = model.sigma * model.well_width
        well_depth = -model.epsilon_over_kB * model.well_depth
        
        r_repulsive = r[r < model.sigma]
        if len(r_repulsive) > 0:
            V_repulsive = np.full_like(r_repulsive, y_max)
            ax.plot(r_repulsive, V_repulsive, '--', 
                   color='#e74c3c', linewidth=2, alpha=0.5)
            
            ax.vlines(x=model.sigma, ymin=well_depth, ymax=y_max,
                     colors='#e74c3c', linestyles='--', alpha=0.7)
        
        r_well = r[(r >= model.sigma) & (r < well_position)]
        V_well = np.full_like(r_well, well_depth)
        ax.plot(r_well, V_well, '-', color='#2E86C1', linewidth=2.5, alpha=0.8)
        
        r_outer = r[r >= well_position]
        V_outer = np.zeros_like(r_outer)
        ax.plot(r_outer, V_outer, '-', color='#2E86C1', linewidth=2.5, alpha=0.8)
        
        ax.vlines(x=well_position, ymin=well_depth, ymax=0,
                 colors='#2E86C1', linestyles='-', alpha=0.8)
    
    else:
        valid_mask = ~np.isinf(V)
        ax.plot(r[valid_mask], V[valid_mask], '-', 
                color='#2E86C1', linewidth=2.5, alpha=0.8)
    
    if not np.isinf(current_V) and current_V < y_max:
        ax.plot(distance, current_V, 'o', color='#e74c3c', 
                markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlim([0.5 * model.sigma, 10.0])
    
    ax.set_xlabel('Distance (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Potential Energy (ε/kB, K)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model.name} Potential', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", 
                     alpha=0.9, linewidth=1.5)
    ax.text(0.98, 0.95, model.equation, 
            transform=ax.transAxes, 
            fontsize=11,
            bbox=bbox_props,
            horizontalalignment='right',
            verticalalignment='top')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    return fig

def main():
    st.title("Molecular Interaction Potential Visualizer")
    
    # Add acknowledgment as a small text below the title
    st.markdown("""
    <div style='font-size: 0.8em; color: #666;'>
    Based on the original <a href='https://github.com/njcorrente/pyPairViz' target='_blank'>pyPairViz</a> by Nick Corrente. 
    This is a web-based implementation using Streamlit.
    </div>
    """, unsafe_allow_html=True)
    
    # Add a visual separator
    st.markdown("---")
    
    # Initialize session state
    if 'current_distance' not in st.session_state:
        st.session_state.current_distance = 4.0

    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        models = {
            "Lennard-Jones": LennardJones,
            "Hard Sphere": HardSphere,
            "Square Well": SquareWell,
            "Sutherland": Sutherland,
            "Morse": MorsePotential,
            "Buckingham": BuckinghamPotential,
            "Yukawa": YukawaPotential,
            "Mie": MiePotential
        }
        
        model_name = st.selectbox(
            "Select Potential Model",
            list(models.keys())
        )
        
        # Base parameters
        epsilon = st.number_input("ε/kB (K)", value=120.0, min_value=0.0)
        sigma = st.number_input("σ (Å)", value=3.4, min_value=0.1)
        
        # Model-specific parameters
        model = models[model_name](epsilon_over_kB=epsilon, sigma=sigma)
        
        if model_name == "Square Well":
            model.well_width = st.slider("Well Width (λ, in σ units)", 1.0, 3.0, 1.5)
            model.well_depth = st.slider("Well Depth (in ε units)", 0.1, 2.0, 1.0)
        elif model_name == "Morse":
            model.a = st.slider("Width parameter (a)", 0.1, 5.0, 1.0)
        elif model_name == "Buckingham":
            model.A = st.number_input("Repulsive strength (A)", value=1000.0)
            model.B = st.slider("Repulsive range (B)", 0.1, 5.0, 2.0)
        elif model_name == "Yukawa":
            model.kappa = st.slider("Screening length (κ)", 0.1, 5.0, 1.0)
        elif model_name == "Mie":
            model.n = st.slider("Repulsive exponent (n)", 6, 18, 12)
            model.m = st.slider("Attractive exponent (m)", 4, 10, 6)
        elif model_name == "Sutherland":
            model.n = st.slider("Power (n)", 6, 18, 12)

    # Main content
    st.header("Model Information")
    st.write(f"**Description:** {model.description}")
    st.write(f"**Equation:** {model.equation}")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Molecule Visualization")
        distance = st.slider("Molecular Distance (Å)", 
                           min_value=2.0, 
                           max_value=10.0, 
                           value=st.session_state.current_distance,
                           step=0.1)
        st.session_state.current_distance = distance
        
        mol_fig = create_molecule_visualization(distance, sigma)
        st.plotly_chart(mol_fig, use_container_width=True)
    
    with col2:
        st.header("Potential Energy Plot")
        r = np.linspace(0.5*sigma, 10.0, 1000)
        V = model.calculate(r)
        current_V = model.calculate(distance)
        
        fig = create_potential_plot(model, r, V, distance, current_V)
        st.pyplot(fig)

if __name__ == "__main__":
    main()