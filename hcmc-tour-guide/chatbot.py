import streamlit as st
from hcmc_travels import agent_executor

st.title('🦜🔗 Quickstart App')

def generate_response(input_text):
    response = agent_executor.invoke({
    'input': input_text
    })

    st.info(response['output'])

with st.form('my_form'):
    text = st.text_area('Enter text:', 'what should I visit in Ho Chi Minh City?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)