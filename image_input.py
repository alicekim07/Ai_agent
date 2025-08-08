import streamlit as st

def get_image():
    """
    사용자로부터 이미지 파일 경로를 입력받아 바이너리 데이터를 반환합니다.
    실제 서비스에서는 업로드/웹캠 등으로 확장 가능.
    """
    path = input("이미지 파일 경로 입력: ")
    with open(path, "rb") as f:
        return f.read()

# Streamlit용 업로드 함수
def get_image_streamlit():
    uploaded_file = st.file_uploader("장난감 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        return uploaded_file.read()
    return None