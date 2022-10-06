import streamlit as st
from utils.intro import intro
from utils.tools import super_resolution, interpolate, sdr2hdr


page_names_to_funcs = {
    "Introduction": intro,
    "Video Super-resolution Compare": super_resolution,
    "Interpolate": interpolate,
    "HDR Imaging": sdr2hdr,
}

demo_name = st.sidebar.selectbox("Choose a Application", page_names_to_funcs.keys())

if demo_name == 'Video Super-resolution Compare' or  demo_name == 'HDR Imaging':
    video_path = 'video'
    page_names_to_funcs[demo_name](video_path=video_path)
elif demo_name == 'Interpolate':
    model_path = 'weight'
    page_names_to_funcs[demo_name](model_path=model_path)
else:
    page_names_to_funcs[demo_name]()