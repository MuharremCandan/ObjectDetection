
import image
import streamlit as st



def main():
    new_title = '<p style="font-size: 50px;">Muharrem Candan [190541055]</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    image.object_detection_image()

if __name__ == '__main__':
    main()



