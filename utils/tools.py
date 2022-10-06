import os
import torch
import mmcv
import cv2
import numpy as np
from copy import deepcopy


def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    # st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    # st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    # st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


def init_posters(video_path=None):
    import cv2
    import os

    if not video_path:
        raise 'Value Error!'
    else:
        posters = list()
        videos = os.listdir(video_path)
        videos = [v for v in videos if v.endswith(".mp4")]

        for v in videos:
            cap = cv2.VideoCapture(os.path.join(video_path, v))
            _, frame = cap.read()
            frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, [240,160])
            posters.append(frame)
            cap.release()
        return posters


def load_model(model, device, model_path=None):
    if not model_path:
        return model

    load_net = torch.load(model_path, map_location=lambda storage, loc:storage)
    load_net = load_net['state_dict']

    choose_key = 'generator'
    for key, value in deepcopy(load_net).items():
        key_list = key.split('.')

        if choose_key in key_list:
            tmp_key = ".".join(key_list[1:])
            load_net[tmp_key] = value
    
        load_net.pop(key)

    model.load_state_dict(load_net, strict=True)
    return model


def super_resolution(video_path=None):
    import os
    import cv2
    from streamlit_image_comparison import image_comparison
    import streamlit as st

    st.write("# Welcome to Super-resolution Demo!")

    # 初始化文件海报
    posters = init_posters(video_path=video_path)
    cols = st.columns(5)

    with cols[0]:
        st.image(posters[0], caption='Clip: Test01')
    
    with cols[1]:
        st.image(posters[1], caption='Clip: Test02')
    
    with cols[2]:
        st.image(posters[2], caption='Clip: Test03')
    
    with cols[3]:
        st.image(posters[3], caption='Clip: Test04')

    with cols[4]:
        st.image(posters[4], caption='Clip: Test05')
    
    # 输入的视频字典
    video_clips = {
        "-": '-1',
        "Clip: Test01": '1',
        "Clip: Test02": '2',
        "Clip: Test03": '3',
        "Clip: Test04": '4',
        "Clip: Test05": '5'
    }
    
    video_name = st.selectbox("Please choose a clip", video_clips.keys())

    if int(video_clips[video_name]) != -1:
        clip_path_lr = os.path.join(video_path, "test0{:d}.mp4".format(int(video_clips[video_name])))
        clip_path_sr = os.path.join(video_path, 'sr', "test0{:d}.mp4".format(int(video_clips[video_name])))

        lr_cap = mmcv.VideoReader(clip_path_lr)
        sr_cap = mmcv.VideoReader(clip_path_sr)

        lr_frame = lr_cap[0]
        sr_frame = sr_cap[0]
        lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
        sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)

        my_slider = st.slider("选择视频帧",0,len(lr_cap)-1,0,1)
        
        if my_slider:
            lr_frame = lr_cap[my_slider]
            sr_frame = sr_cap[my_slider]

            lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
            sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)

        image_comparison(img1=lr_frame, img2 = sr_frame, label1="LR", label2="Ours")


def compute_flow_map(model, lrs_ndarray, device=None):
    h, w, c = lrs_ndarray[0].shape

    lrs_zero_to_one = [v.astype(np.float32) / 255. for v in lrs_ndarray]
    lrs_tensor = [torch.from_numpy(v).permute(2,0,1) for v in lrs_zero_to_one]
    
    
    lrs = torch.cat(lrs_tensor).view(-1, c, h, w).unsqueeze(0)
    
    if device is not None:
        lrs = lrs.to(device)

    _, flow = model(lrs)
    flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
    flow = flow[0]
    flow_map = np.uint8(mmcv.flow2rgb(flow) * 255.)
    return flow_map


def interpolate(model_path = None):
    import streamlit as st
    import numpy as np
    import cv2
    from utils.archs.basicvsr_pp_gauss import BasicVSRPlusPlus_Gauss
    from utils.archs.basicvsr_pp import BasicVSRPlusPlus
    from streamlit_image_comparison import image_comparison

    st.write("# Video Interpolate Demo!")
    frame1 = st.file_uploader("请选择视频帧1") # 选择上传一个文件
    frame2 = st.file_uploader("请选择视频帧2") # 选择上传一个文件

    if frame1:
        file_bytes = np.asarray(bytearray(frame1.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 显示上传图片
        frame1 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        frame1 = cv2.resize(frame1, [240,160])
        st.image(frame1)
        
    if frame2:
        file_bytes = np.asarray(bytearray(frame2.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 显示上传图片
        frame2 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        frame2 = cv2.resize(frame2, [240,160])
        st.image(frame2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    model1 = BasicVSRPlusPlus_Gauss()
    model1 = load_model(model1, device, model_path=os.path.join(model_path, "basicvsr_pp_gauss.pth"))
    model1.to(device)

    model2 = BasicVSRPlusPlus()
    model2 = load_model(model2, device, model_path=os.path.join(model_path, "basicvsr_plusplus.pth"))
    model2.to(device)

    models.append(model2)
    models.append(model1)

    lrs_ndarray = []
    if frame1 is not None and frame2 is not None and models:
        lrs_ndarray.append(frame1)
        lrs_ndarray.append(frame2)
    
        flow_maps = []
        for model in models:
            flow_map = compute_flow_map(model, lrs_ndarray, device)
            flow_maps.append(flow_map)
        
        image_comparison(img1=flow_maps[0], img2=flow_maps[1], label1="BasicVSR++", label2="Ours")    


def sdr2hdr(video_path = None):
    import streamlit as st
    from streamlit_image_comparison import image_comparison

    st.write("# Video HDR Imaging Demo!")

    clip_path_lr = os.path.join(video_path, "test0{:d}.mp4".format(1))
    clip_path_sr = os.path.join(video_path, 'hdr', "test0{:d}.mp4".format(1))

    sdr_cap = mmcv.VideoReader(clip_path_lr)
    hdr_cap = mmcv.VideoReader(clip_path_sr)

    lr_frame = sdr_cap[0]
    sr_frame = hdr_cap[0]
    lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
    sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)

    my_slider = st.slider("选择视频帧",0,len(sdr_cap)-1,0,1)
    if my_slider:
        lr_frame = sdr_cap[my_slider]
        sr_frame = hdr_cap[my_slider]

        lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
        sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_BGR2RGB)

    image_comparison(img1=lr_frame, img2 = sr_frame, label1="SDR", label2="HDRUNet")