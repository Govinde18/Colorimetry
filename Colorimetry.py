import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    st.title("Estimation of concentration of Unknown sample")
    
    st.header("Input Data")
    st.write("Enter the concentration (µg/ml) and optical density values for the standard samples and unknown sample:")
    
    # Create a form to take inputs
    with st.form("input_form"):
        st.write("### Standard Samples")
        # Creating columns for table-like input
        cols = st.columns(3)
        
        # Header row
        cols[1].write("Concentration (µg/ml)")
        cols[2].write("Optical Density")
        
        # Arrays to store inputs
        concentrations = []
        optical_densities = []
        
        for i in range(6):            
            concentration = cols[1].number_input(f"Concentration {i+1}", min_value=0.0, format="%.2f", step=0.1, key=f"conc_{i}")
            optical_density = cols[2].number_input(f"Optical Density {i+1}", min_value=0.0, format="%.3f", step=0.001, key=f"od_{i}")
            
            concentrations.append(concentration)
            optical_densities.append(optical_density)
        
        # Input for the unknown sample
        st.write("### Unknown Sample")
        cols[1].write("To be calculated")
        unknown_optical_density = cols[2].number_input("Optical Density of unknown sample at specific nm:", min_value=0.0, format="%.3f", step=0.001, key="od_6")
        
        # Submit button
        submit = st.form_submit_button("Calculate Concentration")
    
    if submit:
        if len(concentrations) == 6 and len(optical_densities) == 6:
            try:
                # Convert to numpy arrays
                X = np.array(concentrations).reshape(-1, 1)
                y = np.array(optical_densities)
                
                # Perform linear regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict the concentrations for plotting
                y_pred = model.predict(X)
                
                # Calculate the unknown concentration
                unknown_concentration = (unknown_optical_density - model.intercept_) / model.coef_[0]
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                plt.scatter(concentrations, optical_densities, color='blue', label='Samples of known conc', edgecolor='k', s=100, alpha=0.7)
                plt.plot(X, y_pred, color='red', linewidth=2, label='Std graph')
                plt.scatter([unknown_concentration], [unknown_optical_density], color='green', label='Unknown Sample', edgecolor='k', s=200, alpha=0.9, zorder=5)
                
                # Draw the horizontal and vertical lines
                plt.axhline(y=unknown_optical_density, color='gray', linestyle='--')
                plt.axvline(x=unknown_concentration, color='gray', linestyle='--')
                
                # Adding labels, legend, and title
                plt.xlabel("Concentration (µg/ml)")
                plt.ylabel("Optical Density")
                plt.legend()
                plt.title("Colorimetry Standard Curve")
                plt.grid(True)
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(plt)
                
                # Display the result
                st.write(f"Predicted Concentration of the unknown sample: {unknown_concentration:.2f} µg/ml")
            except Exception as e:
                st.error(f"An error occurred during calculation: {e}")
        else:
            st.error("Please enter values for all standard samples.")

if __name__ == "__main__":
    main()
