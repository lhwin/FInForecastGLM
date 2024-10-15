import streamlit as st
import os

def get_file_list(suffix, path):
    """
    获取当前目录所有指定后缀名的文件名列表、绝对路径列表:param suffix:
    :return:文件名列表、绝对路径列表
    """
    input_template_all =[]
    input_template_all_path =[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:tsu
            if os.path.splitext(name)[1] == suffix:
                input_template_all.append(name)
                input_template_all_path.append(os.path.join(root, name))
    return input_template_all, input_template_all_path

input_folder = st.file_uploader("Choose a folder of files", type=["csv", "png", "jpg"], multiple=True)
path = input_folder
_, file_list_xlsx = get_file_list(".xlsx", path)

st.title("数据分析")
st.sidebar.selectbox("选择当前目录加载的文件", file_list_xlsx)