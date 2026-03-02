import openai
import anthropic
import google.generativeai as genai
from core_logic import rag_pipeline
import requests
import os
from dotenv import load_dotenv
import re

load_dotenv()

# fetching api key for LLM interactions
def get_api_key(service_name):
    """Retrieve API key from environment variables based on service name."""
    env_var_name = f"{service_name.upper()}_API_KEY"
    api_key = os.getenv(env_var_name)

    if not api_key:
        raise ValueError(f"API key for {service_name} not found in environment variables.")

    return api_key

# chat history formatting for different LLMs
def format_chat_history(chat_history, family):
    """Format chat history based on LLM family."""
    formatted_history = []
    if len(chat_history) > 0:
        for history in chat_history:
            user_content = history["user"]
            assistant_content = history["assistant"]
            if family == "gemini":
                formatted_history.extend([
                    {"role": "user", "parts": [user_content]},
                    {"role": "model", "parts": [assistant_content]}
                ])

            else:
                formatted_history.extend([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ])
    return formatted_history

# openai llm handler
def handle_openai(context):
    """Handle requests for OpenAI models."""
    if not context["supports_image"] and context.get("image_urls"):
        return "Images are not supported by selected model.", 0
    try:
        openai.api_key = get_api_key("openai")

        messages = format_chat_history(context["chat_history"], "openai") + [
            {"role": "system", "content": context["SYSTEM_PROMPT"]},
            {"role": "assistant", "content": context["phase_instructions"]},
            {"role": "user", "content": context["user_prompt"]}
        ]

        if context["supports_image"] and context["image_urls"]:
            messages.insert(2, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}} for url in
                                                            context["image_urls"]]})

        response = openai.chat.completions.create(
            model=context["model"],
            messages=messages,
            temperature=context["temperature"],
            max_tokens=context["max_tokens"],
            top_p=context["top_p"],
            frequency_penalty=context["frequency_penalty"],
            presence_penalty=context["presence_penalty"]
        )
        input_price = int(getattr(response.usage, 'prompt_tokens', 0)) * context["price_input_token_1M"] / 1000000
        output_price = int(getattr(response.usage, 'completion_tokens', 0)) * context["price_output_token_1M"] / 1000000
        execution_price = input_price + output_price
        return response.choices[0].message.content, execution_price
    except Exception as e:
        return f"Unexpected error while handling OpenAI request: {e}", 0

# claude llm handler
def handle_claude(context):
    """Handle requests for Claude models."""
    if not context["supports_image"] and context.get("image_urls"):
        return "Images are not supported by selected model.", 0
    try:
        client = anthropic.Anthropic(api_key=get_api_key("claude"))

        # Clean up any trailing whitespace in the messages
        messages = format_chat_history(context["chat_history"], "claude") + [
            {"role": "user", "content": [{"type": "text", "text": context["phase_instructions"].strip()}]},
            {"role": "user", "content": [{"type": "text", "text": context["user_prompt"].strip()}]},
        ]

        if context["supports_image"] and context["image_urls"]:
            for image_url in context["image_urls"]:
                # Extract base64 data from the image URL
                base64_data = image_url.split(",")[1]
                mime_type = re.search(r"data:(.*?);base64,", image_url).group(1) if re.search(r"data:(.*?);base64,",
                                                                                              image_url) else None
                # Add image to the messages
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_data
                        }
                    }]
                })
        response = client.messages.create(
            model=context["model"],
            max_tokens=context["max_tokens"],
            temperature=context["temperature"],
            system=f"{context['SYSTEM_PROMPT']}",
            messages=messages
        )
        input_price = int(getattr(response.usage, 'input_tokens', 0)) * context["price_input_token_1M"] / 1000000
        output_price = int(getattr(response.usage, 'output_tokens', 0)) * context["price_output_token_1M"] / 1000000
        execution_price = input_price + output_price
        
        response_text = '\n'.join([block.text for block in response.content if block.type == 'text'])
        return response_text, execution_price
    except Exception as e:
        execution_price = 0
        return f"Unexpected error while handling Claude request: {e}", execution_price

# gemini llm handler
def handle_gemini(context):
    """Handle requests for Gemini models."""
    if not context["supports_image"] and context.get("image_urls"):
        return "Images are not supported by selected model.", 0
    try:
        genai.configure(api_key=get_api_key("google"))

        messages = format_chat_history(context["chat_history"], "gemini") + [
            {"role": "model", "parts": [context["phase_instructions"]]},
            {"role": "user", "parts": [context["user_prompt"]]}
        ]

        if context["supports_image"] and context["image_urls"]:
            for image_url in context["image_urls"]:
                # Add image to the messages
                messages.append({
                    "role": "user",
                    "parts": [image_url]
                })

        chat_session = genai.GenerativeModel(
            model_name=context["model"],
            generation_config= {"temperature": context["temperature"],"top_p": context["top_p"],"max_output_tokens": context["max_tokens"],"response_mime_type":"text/plain"},
            system_instruction=context["SYSTEM_PROMPT"]
        ).start_chat(history=messages)

        response = chat_session.send_message(context["user_prompt"])

        input_price = getattr(response.usage_metadata, 'prompt_token_count', 0) * context["price_input_token_1M"] / 1000000
        output_price = getattr(response.usage_metadata, 'candidates_token_count', 0) * context["price_output_token_1M"] / 1000000
        execution_price = input_price + output_price
        return response.text, execution_price
    except Exception as e:
        return f"Unexpected error while handling Gemini request: {e}", 0

# perplexity handler
def handle_perplexity(context):
    """Handle requests for Perplexity models."""
    if not context["supports_image"] and context.get("image_urls"):
        return "Images are not supported by selected model.", 0
    api_key = get_api_key("perplexity")
    url = "https://api.perplexity.ai/chat/completions"
    execution_price = 0
    # Prepare messages
    messages = [
                   {"role": "system", "content": context["SYSTEM_PROMPT"] + context["phase_instructions"]}
               ] + format_chat_history(context["chat_history"], "perplexity") + [
                   {"role": "user", "content": context["user_prompt"]}
               ]

    # Add image URLs if supported
    if context["supports_image"] and context["image_urls"]:
        for image_url in context["image_urls"]:
            messages.append({
                "role": "user",
                "content": {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            })

    # Prepare payload
    payload = {
        "model": context["model"],
        "messages": messages
    }

    # Prepare headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    # Make the API request
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        response_json = response.json()

        if "usage" in response_json:
            input_price = response_json["usage"]["prompt_tokens"] * context["price_input_token_1M"] / 1000000
            output_price = response_json["usage"]["total_tokens"] * context["price_output_token_1M"] / 1000000
            execution_price = input_price + output_price


        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"], execution_price
        else:
            return "Unexpected response format from Perplexity API.", execution_price

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred while handling Perplexity request: {http_err}", execution_price
    except requests.exceptions.RequestException as req_err:
        return f"Error occurred while making the Perplexity request: {req_err}", execution_price


def rag_handler(context):
    """
    RAG Handler that processes the document, retrieves relevant information,
    and generates a response using the OpenAI language model.
    Falls back to direct LLM response if RAG is not available.

    Args:
    - context: A dictionary containing the file path, user prompt, and LLM configuration.

    Returns:
    - Generated response and cost.
    """
    # Step 1: Extract necessary information from the context
    file_path = context.get("file_path", None)
    user_prompt = context.get("user_prompt", "")

    if not user_prompt:
        raise ValueError("User prompt is required.")

    # If no file path is provided, fall back to direct LLM response
    if not file_path:
        print("No file path provided for RAG. Falling back to direct LLM response.")
        try:
            rag_response, cost = rag_pipeline._fallback_llm_response(
                question=user_prompt,
                template_text=str(context["phase_instructions"]) + " User answer is " + user_prompt
            )
            context["TOTAL_PRICE"] = context.get("TOTAL_PRICE", 0) + (cost if cost else 0)
            return rag_response, cost if cost else 0
        except Exception as e:
            return f"Error during fallback processing: {e}", 0

    # Step 2: Check and store metadata and embeddings if not already present
    metadata_result = rag_pipeline.check_and_store_metadata_and_embeddings(file_path)
    print(f"Metadata check result: {metadata_result}")

    # Step 3: Retrieve relevant documents based on the user's query and generate a response
    try:
        # Call the retrieval and response generation pipeline
        rag_response, cost = rag_pipeline.retrieve_and_generate_response(
            question=user_prompt,
            template_text=str(context["phase_instructions"]) + " User answer is " + user_prompt
        )
        print(f"RAG response: {rag_response}, cost: {cost}")
        # Step 4: Update the context with the cost (if applicable)
        context["TOTAL_PRICE"] = context.get("TOTAL_PRICE", 0) + (cost if cost else 0)
        return rag_response, cost if cost else 0
    except Exception as e:
        print(f"RAG processing failed: {e}. Attempting fallback.")
        # Try fallback if RAG fails
        try:
            rag_response, cost = rag_pipeline._fallback_llm_response(
                question=user_prompt,
                template_text=str(context["phase_instructions"]) + " User answer is " + user_prompt
            )
            context["TOTAL_PRICE"] = context.get("TOTAL_PRICE", 0) + (cost if cost else 0)
            return rag_response, cost if cost else 0
        except Exception as fallback_e:
            return f"Error during RAG and fallback processing: {e} | {fallback_e}", 0


# Mapping of model families to handler functions
HANDLERS = {
    "openai": handle_openai,
    "claude": handle_claude,
    "gemini": handle_gemini,
    "perplexity": handle_perplexity,
    "rag":rag_handler
}
