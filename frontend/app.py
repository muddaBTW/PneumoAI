import streamlit as st
import requests

st.title('Pneumonia Detection AI')

upload_file = st.file_uploader('Upload Chest X-Ray',type=['jpg','png','jpeg'])

if upload_file is not None:

    # send file to backend API
    files = {'file':upload_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

    if response.status_code == 200:
        data = response.json()

        st.subheader('Prediction Result')
        st.write('Disease: ',data['prediction'])
        st.write('Probability: ',round(data['probability'],3))
        st.write('Confidence: ',round(data['confidence'],2), '%')

        if data['prediction'] == 'Pneumonia':
            st.error('Pneumonia Detected')
        else:
            st.success('Normal Lungs')
    
    else:
        st.error('API Error')



