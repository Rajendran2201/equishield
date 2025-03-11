import streamlit as st
import joblib
import re
from bias import bias_replacement_dict, gender_biased_words 

st.set_page_config(page_title="EquiShield", page_icon="🛡️", layout="wide")
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("bias_model.pkl")


def correct_bias_rule_based(text):
    words = text.split()
    corrected_words = []

    for word in words:
        lower_word = word.lower()


        if lower_word in bias_replacement_dict:
            corrected_words.append(bias_replacement_dict[lower_word])
        elif lower_word in gender_biased_words:
            corrected_words.append("(removed)")  
        else:
            corrected_words.append(word)

    corrected_text = " ".join(corrected_words)
    corrected_text = re.sub(r'\s([,.:;!?])', r'\1', corrected_text)  
    return corrected_text


st.title("📰 EquiShield: Bias Detector & Rule-Based Correction")

st.write("Enter a news headline to analyze for gender bias.")

# User Input
user_input = st.text_area("News Headline", "")

if st.button("Analyze & Correct Bias"):
    if user_input.strip():

        processed_input = user_input.lower()
        processed_input = re.sub(r'[^\w\s]', '', processed_input)  # Remove punctuation


        input_vectorized = vectorizer.transform([processed_input])
        probabilities = model.predict_proba(input_vectorized)[0]
        bias_probability = probabilities[1]

        col1, col2 = st.columns(2)

        with col1:
            if bias_probability > 0.5:
                # st.error("🚨 Potential Gender Bias Detected")
                unbiased_text = correct_bias_rule_based(user_input)
                st.success(" Rule-Based Suggested Unbiased Version:")
                st.write(f"**{unbiased_text}**")
            else:
                st.success("No Significant Gender Bias Detected")

        # with col2:
        #     st.metric("Bias Probability", f"{bias_probability:.1%}")

    else:
        st.warning("Please enter a headline to analyze.")
