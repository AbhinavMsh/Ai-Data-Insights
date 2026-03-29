figures = patterns_2_main(patterns, top_n=10)

st.subheader("Correlation Heatmaps")
col1, col2, col3 = st.columns(3)

with col1:
    if figures["pearson_heatmap"]:
        st.plotly_chart(figures["pearson_heatmap"],  use_container_width=True)
        st.caption("Pearson Correlation")
with col2:
    if figures["spearman_heatmap"]:
        st.plotly_chart(figures["spearman_heatmap"], use_container_width=True)
        st.caption("Spearman Correlation")
with col3:
    if figures["cramers_heatmap"]:
        st.plotly_chart(figures["cramers_heatmap"],  use_container_width=True)
        st.caption("Cramér's V Association")

st.subheader("Top Significant Pairs")
col4, col5, col6 = st.columns(3)

with col4:
    if figures["pearson_bar"]:
        st.plotly_chart(figures["pearson_bar"],  use_container_width=True)
with col5:
    if figures["spearman_bar"]:
        st.plotly_chart(figures["spearman_bar"], use_container_width=True)
with col6:
    if figures["cramers_bar"]:
        st.plotly_chart(figures["cramers_bar"],  use_container_width=True)