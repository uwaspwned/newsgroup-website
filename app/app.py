import os

import requests

import gradio as gr
from gradio.themes import Soft

FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://localhost:8000')
PREDICT_URL = f"{FASTAPI_URL}/predict"

API_KEY = "gradio-secret-key-12345" 


def predict_via_api(text: str) -> str:
    """Send a request to the FastAPI backend for sentiment prediction"""

    try:
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }

        response = requests.post(
            PREDICT_URL,
            json={"text": text},
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            category_name = result["category_name"]

            return f"{category_name}"
        elif response.status_code == 403:
            return "Authentication failed. Check API key configuration"
        elif response.status_code == 429:
            return "Rate limit exceeded. Please wait"
        else:
            return f"API error {response.status_code}"
      
    except requests.exceptions.Timeout:
        return "Request timeout - API is slow"
    except requests.exceptions.ConnectionError:
        return "Could not connect to API. Make sure FastAPI is running."
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Your letter",
                placeholder="Write your letter here...",
                lines=5
            )
            submit_btn = gr.Button("Analyze", variant="primary", interactive=True)

        with gr.Column():
            output_text = gr.Textbox(label="Result", lines=5)

    gr.Examples(
        examples=[
            ["We invite you to read the Bible today."],
            ["Hi, have you seen the pre-release of the new Windows?"],
            ["Space is incredible. I wish I could see the stars up close."]
        ],
        inputs=text_input
    )

    submit_btn.click(
        fn=predict_via_api,
        inputs=text_input,
        outputs=output_text
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=Soft())