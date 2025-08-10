import streamlit as st
from PIL import Image
import io

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

def optimize_image_size(image_bytes, max_size=512):
    """이미지 크기를 최적화하여 인코딩 시간을 단축"""
    try:
        # 이미지 열기
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 크기 확인
        width, height = image.size
        
        # 이미 충분히 작으면 그대로 반환
        if width <= max_size and height <= max_size:
            return image_bytes
        
        # 비율 유지하면서 리사이즈
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # 리사이즈
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 바이트로 변환
        output = io.BytesIO()
        resized_image.save(output, format='JPEG', quality=85, optimize=True)
        optimized_bytes = output.getvalue()
        
        print(f"이미지 최적화: {width}x{height} → {new_width}x{new_height}")
        return optimized_bytes
        
    except Exception as e:
        print(f"이미지 최적화 실패: {e}")
        return image_bytes