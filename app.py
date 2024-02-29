import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
from openai import OpenAI

# Initialize the Sentiment Analyzer and download necessary data
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Securely set your OpenAI API key
api_key = "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"  # Use st.secrets in production

def get_sentiment(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'positive', compound_score
    elif compound_score <= -0.05:
        return 'negative', compound_score
    else:
        return 'neutral', compound_score
    
def generate_response_with_chatgpt(input_text, tone):
    client = OpenAI(api_key=api_key)

    # Prepare messages payload including conversation history
    messages = []
    for i, msg in enumerate(st.session_state['messages']):
        # Alternate roles for messages based on their order
        role = "assistant" if i % 2 else "user"
        messages.append({"role": role, "content": msg})
    
    # Add the current user's message
    messages.append({"role": "user", "content": f"Write a really {tone} response to: '{input_text}'"})

    # Request a chat completion using the specified method and model
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    
    # Extract and return the generated response
    response_text = chat_completion.choices[0].message.content
    return response_text

def handle_message_input(user_input):
    # Check if message count exceeds 5
    if st.session_state['message_count'] > 5:
        # Reset session state
        st.session_state['message_count'] = 0
        st.session_state.clear()  # This clears the list but keeps the key in session state
        # Optionally, you can use st.session_state.clear() to remove all session state keys
        st.experimental_rerun()  # Rerun the app to reflect the reset immediately

# Initialize session state variables
if 'message_count' not in st.session_state:
    st.session_state['message_count'] = 0
if 'mood' not in st.session_state:
    st.session_state['mood'] = ''
if 'sentiment_scores' not in st.session_state:
    st.session_state['sentiment_scores'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# UI for setting the mood
st.title("AI Chat Application")
st.caption("This is a tweaked chatbot that uses the OpenAI Chat API. The user sets the mood of the conversation and the AI responds in the selected mood. Set the mood and try to get a respond up to 5 times, to continue the conversation, type in the textbox above RESPOND ")
if 'mood' not in st.session_state or st.session_state['mood'] == '':

    mood_setter = st.text_input("How do you feel?", key="mood_setter")
    if st.button("Set Mood"):
        conversation_tone, compound_score = get_sentiment(mood_setter)
        st.session_state['mood'] = conversation_tone
        st.success(f"Mood set to {conversation_tone}.")


#UI for the conversation
if st.session_state['mood']:
    user_input = st.text_input("Type your message:", key="user_message")
    if st.button("Respond"):
        st.session_state['message_count'] += 1
        st.write("Maxiumum 5 messages per conversation.")
        st.write('message count:', st.session_state['message_count'])

        # Save user message and display AI response
        st.write("latest compound score:")
        _, compound_score = get_sentiment(user_input)  # Correctly capturing the compound 
        st.write(compound_score)
        handle_message_input(user_input)  # Assuming this is for user inputs only
        
        # Append compound score for the user's input
        st.session_state['sentiment_scores'].append(compound_score)
        
        # Generate and display AI response
        response = generate_response_with_chatgpt(user_input, st.session_state['mood'])
        st.session_state['messages'].append(f"You: {user_input}")
        st.session_state['messages'].append(f"AI: {response}")
        
        # Display conversation history
        conversation_history = "\n".join(st.session_state['messages'])
        st.text_area("Conversation History", conversation_history, height=300)
        
        # Display the line chart for sentiment scores
        if st.session_state['sentiment_scores']:
            st.write("Sentiment Score Over Conversation:")
            st.write("0 is neutral, + is positive, and - is negative")
            st.write("zoom out and see where the conversation is going in terms of your mood")
            st.line_chart(st.session_state['sentiment_scores'])

if st.button('Reset Session'):
    st.session_state.clear()
    st.rerun()
