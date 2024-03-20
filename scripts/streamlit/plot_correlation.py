import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/stolosa/Documents/BSC_internship/Local/Proyecto_MASTER/analysis/mutant_analysis/git_repo/scripts")
import CORR_methods as mf

# First of all we need to load our dataframe
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
else:
    st.write("No data selected")
    st.stop()

# Select the variables we want to compare (only numerical variables)
numerical_variables = df.select_dtypes(include="number").columns.tolist()
color_variables = df.select_dtypes(include="object").columns.tolist()

variable1 = st.selectbox("Select a variable", numerical_variables)
variable2 = st.selectbox("Select a second variable", numerical_variables)
color = st.selectbox("Color variable", color_variables)


st.title(f"Correlation analysis between {variable1} and {variable2}")

# Correlation coefficient

R, p_value = mf.correlation_analysis(df[variable1].to_numpy(), df[variable2].to_numpy(), test="spearman")

st.write(f"- Correlation coefficient:{R:.2f}")
st.write(f"- P-value:{p_value:.1e}")




# Display the plot:
fig, ax = plt.subplots()

ax.scatter(x = df[variable1], y = df[variable2], s=5)
ax.set_xlabel(variable1)
ax.set_ylabel(variable2)

fn = "scatter_plot.png"
plt.savefig(fn)

st.pyplot(fig)

with open(fn, "rb") as f:
    st.download_button(
        label="Download image",
        data=f,
        file_name=fn,
        mime="image/png"
)

 



