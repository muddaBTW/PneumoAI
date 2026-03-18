import streamlit as st
import requests

st.set_page_config(page_title="PneumoAI - Medical Assistant", layout="wide")

st.title('Pneumonia Detection AI')

# Initialize session state for chat and prediction persistence
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# Sidebar for reset
with st.sidebar:
    st.subheader("Settings")
    model_options = {
        "Llama 3.3 70B (Stable)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
        "Mixtral 8x7B": "mixtral-8x7b-32768"
    }
    selected_model_name = st.selectbox("Select Vision Model", options=list(model_options.keys()), index=0)
    selected_model_id = model_options[selected_model_name]
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("X-Ray Analysis")
    upload_file = st.file_uploader('Upload Chest X-Ray', type=['jpg', 'png', 'jpeg'])

    if upload_file is not None:
        st.image(upload_file, caption="Uploaded X-ray", use_container_width=True)

        if st.button('Analyze X-Ray', type="primary"):
            # Store image in session state
            st.session_state.image_bytes = upload_file.getvalue()
            # send file to backend API
            files = {'file': st.session_state.image_bytes}
            with st.spinner("Analyzing image..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/predict", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.prediction_data = data
                    else:
                        st.error(f'API Error: {response.status_code}')
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")

    if st.session_state.prediction_data:
        data = st.session_state.prediction_data
        st.markdown("---")
        st.subheader('Analysis Results')
        
        result_col, info_col = st.columns(2)
        with result_col:
            if data['prediction'] == 'Pneumonia':
                st.error(f"**Finding:** {data['prediction']}")
            else:
                st.success(f"**Finding:** {data['prediction']}")
        
        with info_col:
            st.metric("Confidence", f"{data['confidence']:.2f}%")
            st.write(f"Probability Score: {data['probability']:.4f}")

with col2:
    st.subheader("Medical Chat Assistant")
    
    if not st.session_state.prediction_data:
        st.info("Please upload and analyze an X-ray to start the medical conversation.")
    else:
        # Display chat messages from history
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about your results..."):
            # Display user message in chat message container
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    with st.spinner("Consulting Groq Vision..."):
                        # Convert image to base64 from session state
                        import base64
                        img_bytes = st.session_state.image_bytes
                        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

                        chat_payload = {
                            "message": prompt,
                            "prediction_context": st.session_state.prediction_data['prediction'],
                            "confidence_context": st.session_state.prediction_data['confidence'],
                            "image_b64": img_b64,
                            "model_id": selected_model_id
                        }
                        try:
                            chat_res = requests.post("http://127.0.0.1:8000/chat", json=chat_payload)
                            if chat_res.status_code == 200:
                                assistant_response = chat_res.json()["response"]
                                st.markdown(assistant_response)
                                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            else:
                                st.error("Chat API Error")
                        except Exception as e:
                            st.error(f"Chat Connection Error: {str(e)}")
            st.rerun()



