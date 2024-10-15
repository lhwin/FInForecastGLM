import streamlit as st
import numpy as np
import plotly.figure_factory as ff

x1 = np.random.randn(200)-2
x2 = np.random.randn(200)
x3 = np.random.randn(200)+2

hist_data = [x1, x2, x3]

labels = ["g1", "g2", "g3"]
fig = ff.create_distplot(hist_data, labels, bin_size=[0.1, 0.25, 0.25])

st.plotly_chart(fig, use_container_width=True)