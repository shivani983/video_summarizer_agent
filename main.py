import streamlit as st
from phi.agent import Agent

from phi.model.google import Gemini

from phi.tools.duckduckgo import DuckDuckGo

from google.generativeai import upload_file, get_file

import google.generativeai as genai
import time
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key = API_KEY)


st.set_page_config(
    page_title = "Video Summarizer Agent",
    page_icon = "📸",
    layout = "wide"
) 

st.title("video summarizer agent 🎥")
st.header("this application is powered by Gemini")


def initialize_agent():
    return Agent(
        name = "Video Summarizer Agent",
        model = Gemini(id = "gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown = True
    )

multimodel_agent = initialize_agent()

video_file = st.file_uploader(
    "upload your video you want to summarize", type = ['mp4', 'mov', 'avi'], help = "upload your video for analysis to be done by AI"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path,format = "video/mp4", start_time = 0)

    user_query = st.text_area(
        "What key insights are you hoping to uncover from this video?",
        placeholder = "Ask anything about your video! Simply upload your video and type your query to get insights instantly.🚀🎥",
        help = "Need help? Simply upload your video, type your query, and let our AI provide insights instantly. For best results, ensure your video is clear and relevant to your question.💡🎥"
    )  

    if st.button("Analyze Video",key ="analyze_video_button"):
        if not user_query:
            st.warning("Upload your video and ask any question—our AI will analyze and provide insights instantly!🚀🎥")
        else:
            try:
                with st.spinner("Analyzing your video... Please wait while we fetch insights!🚀🎥"):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "Analyzing":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)


                    analysis_prompt = (
                            f""""Analyze the provided video and extract key insights based on the {user_query}. 
                            Summarize important topics, detect relevant entities, and provide clear, concise responses.
                            Ensure accuracy and relevance to the query while maintaining a user-friendly explanation."""
                        )

                    response = multimodel_agent.run(analysis_prompt, videos=[processed_video])

                    if response:
                        st.subheader("Analysis Result")
                        st.markdown(response.content)
                    else:
                        st.error("Analysis failed. No valid response received.")

            except Exception as error:
                st.error(f" An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
    else:
        st.info("upload a video file to begin analysis")            

                        
st.markdown(
    """
    <style>
    .video-frame {
        width: 200px;
        height: 200px;
        overflow: hidden;
        border: 2px solid black;
        border-radius: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

