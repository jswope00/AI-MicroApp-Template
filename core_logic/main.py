import re
import base64
import mimetypes
import os
import streamlit as st
from streamlit import _bottom
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.let_it_rain import rain
from core_logic.handlers import HANDLERS
from core_logic.llm_config import LLM_CONFIG
from core_logic.data_storage import StorageManager
from datetime import datetime
import pandas as pd

# Function to evaluate conditional logic
def evaluate_conditions(user_input, condition):
    """
    Evaluates whether the 'user_input' meets the specified 'condition'.
    Supports logical operators like $and, $or, $not, and comparison operators.
    """
    if "$and" in condition:
        return all(evaluate_conditions(user_input, sub_condition) for sub_condition in condition["$and"])
    elif "$or" in condition:
        return any(evaluate_conditions(user_input, sub_condition) for sub_condition in condition["$or"])
    elif "$not" in condition:
        return not evaluate_conditions(user_input, condition["$not"])

    for key, value in condition.items():
        if isinstance(value, dict):
            operator, condition_value = next(iter(value.items()))
            user_value = user_input.get(key)

            if operator == "$gt" and not user_value > condition_value:
                return False
            elif operator == "$lt" and not user_value < condition_value:
                return False
            elif operator == "$gte" and not user_value >= condition_value:
                return False
            elif operator == "$lte" and not user_value <= condition_value:
                return False
            elif operator == "$eq" and not user_value == condition_value:
                return False
            elif operator == "$ne" and not user_value != condition_value:
                return False
            elif operator == "$in" and user_value not in condition_value:
                return False
            elif operator == "$nin" and user_value in condition_value:
                return False
        else:
            if isinstance(value, list):
                if user_input.get(key) not in value:
                    return False
            else:
                if user_input.get(key) != value:
                    return False
    return True

# Function to build input fields based on configuration
def build_field(phase_name, fields, user_input, phases, system_prompt):
    """
    Builds the input fields for a given phase based on the 'fields' configuration.
    Checks for 'showIf' conditions before displaying fields.
    """
    

    function_map = {
        "text_input": st.text_input,
        "text_area": st.text_area,
        "warning": st.warning,
        "button": st.button,
        "radio": st.radio,
        "markdown": st.markdown,
        "selectbox": st.selectbox,
        "checkbox": st.checkbox,
        "slider": st.slider,
        "number_input": st.number_input,
        "image": st.image,
        "file_uploader": st.file_uploader,
        "chat_input": st.chat_input
    }

    for field_key, field in fields.items():
        # Check showIf conditions
        if 'showIf' in field:
            condition = field['showIf']
            if not evaluate_conditions(user_input, condition):
                continue
        field_type = field.get("type", "")
        field_label = field.get("label", "")
        field_body = field.get("body", "")
        field_value = field.get("value", "")
        field_index = field.get("index", None)
        field_max_chars = field.get("max_chars", None)
        field_help = field.get("help", "")
        field_on_click = field.get("on_click", None)
        field_options = field.get("options", [])
        field_horizontal = field.get("horizontal", False)
        field_min_value = field.get("min_value", None)
        field_max_value = field.get("max_value", None)
        field_step = field.get("step", None)
        field_height = field.get("height", None)
        field_unsafe_html = field.get("unsafe_allow_html", False)
        field_placeholder = field.get("placeholder", "")
        field_image = field.get("image", "")
        field_width = field.get("width", None)
        field_caption = field.get("caption", "")
        field_allowed_files = field.get("allowed_files", None)
        field_multiple_files = field.get("multiple_files", False)
        field_label_visibility = field.get("label_visibility", None)
        field_initial_assistant_message = field.get("initial_assistant_message", "")

        kwargs = {}
        if field_label:
            kwargs['label'] = field_label
        if field_body:
            kwargs['body'] = field_body
        if field_value:
            kwargs['value'] = field_value
        if field_index:
            kwargs['index'] = field_index
        if field_options:
            kwargs['options'] = field_options
        if field_max_chars:
            kwargs['max_chars'] = field_max_chars
        if field_help:
            kwargs['help'] = field_help
        if field_on_click:
            kwargs['on_click'] = field_on_click
        if field_horizontal:
            kwargs['horizontal'] = field_horizontal
        if field_min_value:
            kwargs['min_value'] = field_min_value
        if field_max_value:
            kwargs['max_value'] = field_max_value
        if field_step:
            kwargs['step'] = field_step
        if field_height:
            kwargs['height'] = field_height
        if field_unsafe_html:
            kwargs['unsafe_allow_html'] = field_unsafe_html
        if field_placeholder:
            kwargs['placeholder'] = field_placeholder
        if field_image:
            kwargs['image'] = field_image
        if field_width:
            kwargs['width'] = field_width
        if field_caption:
            kwargs['caption'] = field_caption
        if field_allowed_files:
            kwargs['type'] = field_allowed_files
        if field_multiple_files:
            kwargs['accept_multiple_files'] = field_multiple_files
        if field_label_visibility:
            kwargs['label_visibility'] = field_label_visibility
        if field_initial_assistant_message:
            kwargs['initial_assistant_message'] = field_initial_assistant_message
        key = f"{phase_name}_phase_status"

        # If the user has already answered this question:
        if key in st.session_state and st.session_state[key]:
            # Write their answer
            if f"{phase_name}_user_input_{field_key}" in st.session_state:
                if field_type != "selectbox":
                    kwargs['value'] = st.session_state[f"{phase_name}_user_input_{field_key}"]
                kwargs['disabled'] = True

        my_input_function = function_map[field_type]

        #Render a chat input, with chat history if it exists
        if field_type == "chat_input":
            handle_chat_input(field_key, kwargs, user_input, phase_name, phases, system_prompt)
            continue

        with stylable_container(
                key=f"st_label_{phase_name}_{field_key}_{id(field)}",
                css_styles="""
                    label p {
                        font-weight: bold;
                        font-size: 16px;
                    }

                    div[role="radiogroup"] label p{
                        font-weight: unset !important;
                        font-size: unset !important;
                    }
                    """,
        ):
            user_input[field_key] = my_input_function(**kwargs)

# Function to execute LLM completions
def execute_llm_completions(SYSTEM_PROMPT,selected_llm, phase_instructions, user_prompt, image_urls=None):
    """
    Executes LLM completions using the selected model.
    """
    if selected_llm not in LLM_CONFIG:
        raise ValueError(f"Selected model '{selected_llm}' not found in configuration.")

    model_config = LLM_CONFIG[selected_llm].copy()  # Create a copy of the base config
    
    # Apply any overrides from the app config
    if LLM_CONFIG_OVERRIDE:
        model_config.update(LLM_CONFIG_OVERRIDE)
    
    family = model_config["family"]
    chat_history = st.session_state["chat_history"]

    context = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "phase_instructions": phase_instructions,
        "user_prompt": user_prompt,
        "supports_image": model_config["supports_image"],
        "image_urls": image_urls,
        "model": model_config["model"],
        "max_tokens": model_config["max_tokens"],
        "temperature": model_config["temperature"],
        "top_p": model_config["top_p"],
        "frequency_penalty": model_config.get("frequency_penalty", 0),
        "presence_penalty": model_config.get("presence_penalty", 0),
        "price_input_token_1M": model_config["price_input_token_1M"],
        "price_output_token_1M": model_config["price_output_token_1M"],
        "TOTAL_PRICE": 0,
        "chat_history": chat_history,
        "RAG_IMPLEMENTATION": RAG_IMPLEMENTATION if 'RAG_IMPLEMENTATION' in locals() else False,
        "file_path": "rag_docs/" + SOURCE_DOCUMENT if 'SOURCE_DOCUMENT' in locals() else None,
    }

    handler = HANDLERS.get(family)
    if handler:
        try:
            result = handler(context)
            store_llm_completions(context, result)  
            return result
        except Exception as e:
            raise RuntimeError(f"Error in handling the LLM request: {e}")
    else:
        raise NotImplementedError(f"No handler implemented for model family '{family}'")
    return result


def store_llm_completions(context, result):
    """
    Stores LLM completion data in configured storage with error handling.
    """
    try:
        # Get the existing storage instance
        storage = StorageManager.get_storage()
        
        # Validate result tuple
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Result must be a tuple of (response, execution_price)")
            
        response, execution_price = result
        
        # Validate required context fields
        required_fields = ['phase_instructions', 'user_prompt', 'model']
        missing_fields = [field for field in required_fields if field not in context]
        if missing_fields:
            raise KeyError(f"Missing required context fields: {', '.join(missing_fields)}")
        
        # Create new row with validation
        new_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'APP_TITLE': [APP_TITLE],
            'Phase Instructions': [str(context['phase_instructions'])],
            'User Prompt': [str(context['user_prompt'])],
            'LLM Response': [str(response)],
            'Run Cost': [float(execution_price)],
            'model': [str(context['model'])],
        })
            
        storage.post_runs_data(new_row)
        
        return True
        
    except Exception as e:
        st.error(f"Error in store_llm_completions: {str(e)}")
        print(f"Error in store_llm_completions: {str(e)}")  # For logging
        return None

# Function to apply conditional logic to prompts
def prompt_conditionals(user_input, phase_name=None, phases=None):
    """
    Applies conditional logic to determine the correct prompt based on 'user_input' and 'phase_name'.
    Uses the provided 'phases' to retrieve phase-specific information.
    """
    phase = phases[phase_name]
    if isinstance(phase["user_prompt"], str):
        base_prompt = phase["user_prompt"]
    else:
        additional_prompts = []
        for item in phase["user_prompt"]:
            condition_clause = item["condition"]
            if evaluate_conditions(user_input, condition_clause):
                additional_prompts.append(item["prompt"])
        base_prompt = "\n".join(additional_prompts)
    return base_prompt


# Function to format user prompt with provided inputs
def format_user_prompt(prompt, user_input, phase_name=None, phases=None):
    """
    Formats the 'prompt' using the provided 'user_input' and applies any conditional logic.
    'phases' is required to access phase-specific data.
    Special handling for chat_input fields to include entire chat history.
    """
    try:
        format_dict = {}
        field_types = {}
        prompt = prompt_conditionals(user_input, phase_name, phases)
        
        # Get field types from phase configuration
        if phase_name and phases and phase_name in phases:
            field_types = {
                field_key: field_config.get('type')
                for field_key, field_config in phases[phase_name]['fields'].items()
            }
        
        # Create formatting dictionary
        for key in re.findall(r'{(\w+)}', prompt):
            if key in field_types and field_types[key] == 'chat_input':
                # For chat_input fields, use the entire message history
                chat_messages = st.session_state.get(f"messages_{key}", [])
                # Format chat history as a string
                chat_history = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in chat_messages
                ])
                format_dict[key] = chat_history
            else:
                # For other fields, use the regular user input
                format_dict[key] = user_input.get(key, '') or ''
        
        formatted_user_prompt = prompt.format(**format_dict)
        return formatted_user_prompt
        
    except KeyError as e:
        print(f"KeyError in format_user_prompt: Missing key {e}")
        print("Prompt:", prompt)
        print("User input:", user_input)
        print("Format dict:", format_dict)
        return prompt  # Return unformatted prompt to avoid crashing
    
    except Exception as e:
        print(f"Error occurred in format_user_prompt: {e}")
        return prompt  # As a fallback, return the unformatted prompt


# Function to store session state data
def st_store(input, phase_name, phase_key, field_key=""):
    """
    Stores input data in the session state with keys generated from phase and field names.
    """
    if field_key:
        key = f"{phase_name}_{field_key}_{phase_key}"
    else:
        key = f"{phase_name}_{phase_key}"
    st.session_state[key] = input

# Function to build scoring instructions
def build_scoring_instructions(rubric):
    """
    Builds scoring instructions based on the provided rubric for AI scoring.
    """
    scoring_instructions = f"""
        Please score the user's previous response based on the following rubric: \n{rubric}
        \n\nPlease output your response as JSON, using this format: '{{{{ "[criteria 1]": "[score 1]", "[criteria 2]": "[score 2]", "total": "[total score]" }}}}'
        """
    return scoring_instructions

# Function to extract score from AI response
def extract_score(text):
    """
    Extracts the total score from the AI response text.
    """
    pattern = r'"total":\s*"?(\d+)"?'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    else:
        return 0

# Function to check if the score meets the minimum requirement
def check_score(PHASES,PHASE_NAME):
    """
    Checks if the AI score meets the minimum score requirement for the phase.
    """
    score = st.session_state[f"{PHASE_NAME}_ai_score"]
    try:
        if score >= PHASES[PHASE_NAME]["minimum_score"]:
            st.session_state[f"{PHASE_NAME}_phase_status"] = True
            return True
        else:
            st.session_state[f"{PHASE_NAME}_phase_status"] = False
            return False
    except:
        st.session_state[f"{PHASE_NAME}_phase_status"] = False
        return False

# Function to skip the current phase
def skip_phase(PHASE_NAME, phases, user_input, No_Submit=False):
    """
    Skips the current phase, optionally without submitting data.
    """
    phase_fields = phases[PHASE_NAME]["fields"]  # Access fields from the passed phases argument
    for field_key in phase_fields:
        st_store(user_input[field_key], PHASE_NAME, "user_input", field_key)
    if not No_Submit:
        st.session_state[f"{PHASE_NAME}_ai_response"] = "This phase was skipped."
    st.session_state[f"{PHASE_NAME}_phase_status"] = True
    st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(phases) - 1)


# Function to display a celebration effect
def celebration():
    """
    Displays a celebration effect using falling emojis.
    """
    rain(
        emoji="🥳",
        font_size=54,
        falling_speed=5,
        animation_length=1,
    )

# Function to find image URLs for uploaded app_images
def find_image_urls(user_input,fields):
    """
    Extracts and encodes image URLs from file uploads in the form fields.
    """
    image_urls = []
    for key, value in fields.items():
        if 'decorative' in value and value['decorative']:
            continue
        if 'image' in value:
            image_urls.append(value['image'])
        if 'file_uploader' in value.values():
            uploaded_files = user_input[key]
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
            for uploaded_file in uploaded_files:
                if uploaded_file:
                    file_content = uploaded_file.read()
                    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
                    if not mime_type:
                        mime_type = 'application/octet-stream'
                    base64_encoded_content = base64.b64encode(file_content).decode('utf-8')
                    image_url = f"data:{mime_type};base64,{base64_encoded_content}"
                    image_urls.append(image_url)
    return image_urls

def handle_chat_history(user_input, ai_response, phase_instructions = None,image_urls = None):
    """
    Handles the chat history for a phase, including the assistant instructions.
    """

    # Create a single chat history entry that includes all information
    chat_history_entry = {
        "user": user_input,
        "assistant": ai_response
    }
    
    # Add phase instructions if provided
    if phase_instructions:
        chat_history_entry["assistant_instructions"] = phase_instructions
    
    # Add image URLs if provided
    if image_urls:
        chat_history_entry["app_images"] = image_urls
    
    # Append the single entry to chat history
    st.session_state['chat_history'].append(chat_history_entry)

def handle_chat_input(field_key, kwargs, user_input, phase_name, phases, system_prompt):
    """
    Handles the chat input field type, including message history and LLM interactions.
    """
    selected_llm = st.session_state.get("selected_llm", "openai")  # Default to openai if not set

    # Initialize messages for this specific field
    initial_message = kwargs.pop("initial_assistant_message", "")  # Remove from kwargs
    if f"messages_{field_key}" not in st.session_state:
        st.session_state[f"messages_{field_key}"] = [{"role": "assistant", "content": initial_message}]

    # Get max messages from field configuration, default to 20 if not specified
    phase = phases[phase_name]
    field_config = phase["fields"][field_key]
    max_messages = field_config.get("max_messages", 50)

    # Calculate remaining messages (divide by 2 since each exchange counts as 2 messages)
    current_messages = len(st.session_state[f"messages_{field_key}"]) // 2
    remaining_messages = max_messages - current_messages
    st.write(f"Messages remaining: {remaining_messages}/{max_messages}")

    # Display existing messages
    for message in st.session_state[f"messages_{field_key}"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Check if message limit is reached. Multiplied by 2 to account for user and assistant messages.
    if len(st.session_state[f"messages_{field_key}"]) >= max_messages * 2:
        st.info(
            f"""Notice: The maximum message limit ({max_messages} messages) for this conversation has been reached. 
            Please proceed to the next phase."""
        )
        return

    # Check if this phase is completed
    phase_completed = st.session_state.get(f"{phase_name}_phase_completed", False)
    
    # Only render the chat input if the phase isn't completed and message limit isn't reached
    if not phase_completed and not len(st.session_state[f"messages_{field_key}"]) >= max_messages * 2:
        
        user_input[field_key] = st.chat_input(**kwargs)
        
        if user_input[field_key]:  # If there's a new message
            #Clear any old end-of-phase responses. Usually because the user did not pass
            if f"{phase_name}_ai_response" in st.session_state:
                del st.session_state[f"{phase_name}_ai_response"]
            #Clear any old end-of-phase scores. Usually because the user did not pass
            if f"{phase_name}_ai_score_debug" in st.session_state:
                del st.session_state[f"{phase_name}_ai_score_debug"]
            #Clear any old end-of-phase errors. Usually because the user did not pass
            if f"{phase_name}_error_message" in st.session_state:
                del st.session_state[f"{phase_name}_error_message"]
            #Clear any old end-of-phase warnings. Usually because the user did not pass
            if f"{phase_name}_warning_message" in st.session_state:
                del st.session_state[f"{phase_name}_warning_message"]

            # Display user message            
            with st.chat_message("user"):
                st.markdown(user_input[field_key])
            st.session_state[f"messages_{field_key}"].append({"role": "user", "content": user_input[field_key]})
            
            # Get phase configuration
            phase_instructions = phase.get("phase_instructions", "")
            
            # Get AI response
            ai_response, execution_price = execute_llm_completions(
                system_prompt,
                selected_llm,
                phase_instructions, 
                user_input[field_key]
            )
            st.session_state['TOTAL_PRICE'] += execution_price
            
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(ai_response)
            st.session_state[f"messages_{field_key}"].append({"role": "assistant", "content": ai_response})
            
            # Add to chat history
            handle_chat_history(user_input[field_key], ai_response, phase_instructions)

            
            # Check if we've reached the message limit after this exchange
            #if len(st.session_state[f"messages_{field_key}"]) >= max_messages * 2:
            #    st.session_state[f"{phase_name}_phase_completed"] = True
            
            # Force a rerun to update the chat
            st.rerun()

def handle_submission(PHASE_NAME, PHASE_DICT, fields, user_input, formatted_user_prompt, selected_llm, SYSTEM_PROMPT, PHASES):
    """
    Handles the submission logic for a phase, including AI responses and scoring.
    Returns True if the phase should advance, False otherwise.
    """
    for field_key, field in fields.items():
        st_store(user_input.get(field_key, ""), PHASE_NAME, "user_input", field_key)

    phase_instructions = PHASE_DICT.get("phase_instructions", "")
    image_urls = find_image_urls(user_input, PHASE_DICT.get('fields', {}))

    if PHASE_DICT.get("ai_response", True):
        if PHASE_DICT.get("scored_phase", False):
            if "rubric" in PHASE_DICT:
                # First, provide feedback on the user's response
                ai_feedback, execution_price = execute_llm_completions(SYSTEM_PROMPT, selected_llm, phase_instructions, formatted_user_prompt, image_urls)
                st.session_state['TOTAL_PRICE'] += execution_price
                st.info(body=ai_feedback, icon="🤖")
                
                # Second, provide a score based on the rubric
                scoring_instructions = build_scoring_instructions(PHASE_DICT["rubric"])
                ai_score, score_price = execute_llm_completions("You review the previous conversation and provide a score based on a rubric. You always provide your output in JSON format.", selected_llm, scoring_instructions, formatted_user_prompt)
                st.session_state['TOTAL_PRICE'] += score_price
                st.info(ai_score, icon="🤖")
                
                # Store the feedback and score
                st_store(ai_feedback, PHASE_NAME, "ai_response")
                st_store(ai_score, PHASE_NAME, "ai_score_debug")
                score = extract_score(ai_score)
                st_store(score, PHASE_NAME, "ai_score")

                # Add to chat history
                handle_chat_history(formatted_user_prompt, ai_feedback, phase_instructions, image_urls)

                st.session_state["ai_score"] = ai_score
                st.session_state['score'] = score
                
                if check_score(PHASES, PHASE_NAME):
                    st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
                    st.session_state[f"{PHASE_NAME}_warning_message"] = None
                    st.session_state[f"{PHASE_NAME}_error_message"] = None
                    st.session_state[f"{PHASE_NAME}_phase_completed"] = True
                    return True
                else:
                    if f"messages_{field_key}" in st.session_state:
                        del st.session_state[f"messages_{field_key}"]
                    st_store("You haven't passed. Please try again.", PHASE_NAME, "warning_message")
                    return False
            else:
                st_store("You need to include a rubric for a scored phase", PHASE_NAME, "error_message")
                return False
        else:
            ai_feedback, execution_price = execute_llm_completions(SYSTEM_PROMPT, selected_llm, phase_instructions, formatted_user_prompt, image_urls)
            st_store(ai_feedback, PHASE_NAME, "ai_response")
            st.session_state['TOTAL_PRICE'] += execution_price
            

            # Add to chat history
            handle_chat_history(formatted_user_prompt, ai_feedback, phase_instructions, image_urls)
            
            st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
            st.session_state[f"{PHASE_NAME}_phase_completed"] = True
            return True
    else:
        res_box = st.info(body="", icon="🤖")
        result = ""
        hard_coded_message = PHASE_DICT.get('custom_response', None)
        hard_coded_message = format_user_prompt(hard_coded_message, user_input, PHASE_NAME, PHASES)
        for char in hard_coded_message:
            result += char
            res_box.info(body=result, icon="🤖")
        st.session_state[f"{PHASE_NAME}_ai_response"] = hard_coded_message

        # Add to chat history
        handle_chat_history(formatted_user_prompt, hard_coded_message, phase_instructions, image_urls)

        st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
        st.session_state[f"{PHASE_NAME}_phase_completed"] = True
        return True

# Main function to run the application
def main(config):
    """
    The main entry point for the Streamlit application. Handles page setup, form generation,
    prompt processing, and interaction with LLM for responses.
    """



    # Dynamically get configurations from globals
    PAGE_CONFIG = config.get('PAGE_CONFIG',{})
    SIDEBAR_HIDDEN = config.get('SIDEBAR_HIDDEN', True)
    DISPLAY_COST = config.get('DISPLAY_COST', False)
    global APP_TITLE
    APP_TITLE = config.get('APP_TITLE',"Default Title")
    APP_INTRO = config.get('APP_INTRO', "")
    APP_HOW_IT_WORKS = config.get('APP_HOW_IT_WORKS',"")
    SHARED_ASSET = config.get('SHARED_ASSET', None)
    HTML_BUTTON = config.get('HTML_BUTTON', None)
    PHASES = config.get('PHASES', {"phase1":{"name":"default phase"}})
    COMPLETION_MESSAGE = config.get('COMPLETION_MESSAGE', 'Process completed successfully.')
    COMPLETION_CELEBRATION = config.get('COMPLETION_CELEBRATION', False)
    LLM_CONFIGURATIONS = LLM_CONFIG
    global LLM_CONFIG_OVERRIDE
    LLM_CONFIG_OVERRIDE = config.get('LLM_CONFIG_OVERRIDE', {})
    global GSHEETS_URL_OVERRIDE
    GSHEETS_URL_OVERRIDE = config.get('GSHEETS_URL_OVERRIDE', None)
    global GSHEETS_WORKSHEET_OVERRIDE
    GSHEETS_WORKSHEET_OVERRIDE = config.get('GSHEETS_WORKSHEET_OVERRIDE', "Sheet1")
    PREFERRED_LLM = config.get('PREFERRED_LLM', 'openai')
    SYSTEM_PROMPT = config.get('SYSTEM_PROMPT', '')

    # Apply the page configuration
    if PAGE_CONFIG:
        st.set_page_config(
            page_title=PAGE_CONFIG.get("page_title", "AI MicroApps"),
            page_icon=PAGE_CONFIG.get("page_icon", "🤖"),
            layout=PAGE_CONFIG.get("layout", "wide"),
            initial_sidebar_state=PAGE_CONFIG.get("initial_sidebar_state", "collapsed")
        )

    # Initialize storage once
    StorageManager.initialize(config)


    # Optionally hide the sidebar
    if SIDEBAR_HIDDEN:
        hide_sidebar_style = """
                <style>
                    [data-testid="stSidebar"] {display: none;}
                    [data-testid="stSidebarCollapsedControl"] {display: none;}
                </style>
            """
        st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    # Select template from the sidebar
    selected_template = APP_TITLE


    if "template" not in st.session_state or st.session_state.template != selected_template:
        st.session_state.template = selected_template
        st.query_params["template"] = selected_template
        # Clear all session state variables
        keys_to_keep = ['template']  # Add any other keys you want to preserve
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state['additional_prompt'] = ""
        st.session_state['chat_history'] = []
        st.session_state['CURRENT_PHASE'] = 0
        st.session_state['TOTAL_PRICE'] = 0

        st.rerun()

    user_input = {}

    image_urls = []
    if 'TOTAL_PRICE' not in st.session_state:
        st.session_state['TOTAL_PRICE'] = 0

    # Handle sidebar and model selection
    with st.sidebar:
        llm_options = list(LLM_CONFIGURATIONS.keys())
        llm_index = llm_options.index(PREFERRED_LLM) if PREFERRED_LLM in llm_options else 0

        selected_llm = st.selectbox("Select Language Model", options=LLM_CONFIGURATIONS.keys(), index=llm_index,
                                    key="selected_llm")

        initial_config = LLM_CONFIGURATIONS[selected_llm].copy()

        if LLM_CONFIG_OVERRIDE:
            initial_config.update(LLM_CONFIG_OVERRIDE)
        
        st.session_state['llm_config'] = {
            "model": initial_config["model"],
            "temperature": st.slider("Temperature", min_value=0.0, max_value=1.0,
                                     value=float(initial_config.get("temperature", 1.0)), step=0.01),
            "max_tokens": st.slider("Max Tokens", min_value=50, max_value=10000,
                                    value=int(initial_config.get("max_tokens", 1000)), step=50),
            "top_p": st.slider("Top P", min_value=0.0, max_value=1.0, value=float(initial_config.get("top_p", 1.0)),
                               step=0.1),
            "frequency_penalty": st.slider("Frequency Penalty", min_value=0.0, max_value=1.0,
                                           value=float(initial_config.get("frequency_penalty", 0.0)), step=0.01),
            "presence_penalty": st.slider("Presence Penalty", min_value=0.0, max_value=1.0,
                                          value=float(initial_config.get("presence_penalty", 0.0)), step=0.01),
            "price_input_token_1M": st.number_input("Input Token Price 1M",
                                                    value=initial_config.get("price_input_token_1M", 0)),
            "price_output_token_1M": st.number_input("Output Token Price 1M",
                                                     value=initial_config.get("price_output_token_1M", 0))
        }

        if DISPLAY_COST:
            st.write("Price: ${:.6f}".format(st.session_state['TOTAL_PRICE']))

        # Display chat history in the sidebar
        st.subheader("Chat History")
        for history in st.session_state['chat_history']:
            if 'assistant_instructions' in history:
                with st.chat_message("assistant_instructions"):
                    st.write(history['assistant_instructions'])

            with st.chat_message("user"):
                st.write(history['user'])
                if 'app_images' in history:
                    for image in history['app_images']:
                        st.image(image)
            
            with st.chat_message("assistant"):
                st.write(history['assistant'])
    


    # Main content rendering
    if 'CURRENT_PHASE' not in st.session_state:
        st.session_state['CURRENT_PHASE'] = 0

    st.title(APP_TITLE)
    st.markdown(APP_INTRO)

    # Optionally show "How it Works" section
    if APP_HOW_IT_WORKS:
        with st.expander("Learn how this works", expanded=False):
            st.markdown(APP_HOW_IT_WORKS)

    # Optional asset download
    if SHARED_ASSET:
        with open(SHARED_ASSET["path"], "rb") as asset_file:
            st.download_button(label=SHARED_ASSET["button_text"],
                               data=asset_file,
                               file_name=SHARED_ASSET["name"],
                               mime="application/octet-stream")

    if HTML_BUTTON:
        st.link_button(label=HTML_BUTTON["button_text"], url=HTML_BUTTON["url"])

    

    # Phase rendering and logic
    i = 0
    while i <= st.session_state['CURRENT_PHASE']:
        submit_button = False
        skip_button = False
        final_phase_name = list(PHASES.keys())[-1]
        final_key = f"{final_phase_name}_ai_response"

        PHASE_NAME = list(PHASES.keys())[i]
        PHASE_DICT = PHASES[PHASE_NAME]
        fields = PHASE_DICT["fields"]

        st.write(f"#### Phase {i + 1}: {PHASE_DICT['name']}")

        build_field(PHASE_NAME, fields, user_input, PHASES, SYSTEM_PROMPT)

        key = f"{PHASE_NAME}_phase_status"
        user_prompt_template = PHASE_DICT.get("user_prompt", "")
        if PHASE_DICT.get("show_prompt", False):
            with st.expander("View/edit full prompt"):
                formatted_user_prompt = st.text_area(
                    label="Prompt",
                    height=100,
                    max_chars=50000,
                    value=format_user_prompt(user_prompt_template, user_input, PHASE_NAME,PHASES),
                    disabled=PHASE_DICT.get("read_only_prompt", False)
                )
        else:
            formatted_user_prompt = format_user_prompt(user_prompt_template, user_input, PHASE_NAME,PHASES)

        if PHASE_DICT.get("no_submission", False):
            if key not in st.session_state:
                st.session_state[key] = True
                st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
                st.session_state[f"{PHASE_NAME}_phase_completed"] = True
                st.rerun()

        if key not in st.session_state:
            st.session_state[key] = False

        # Check if current phase has a chat_input field
        has_chat_input = any(field.get('type') == 'chat_input' 
                           for field in PHASE_DICT['fields'].values())

        if not st.session_state.get(f"{PHASE_NAME}_phase_completed", False):
            # Use bottom container for chat input phases, regular container otherwise
            container_class = _bottom.container() if has_chat_input else st.container()
            default_button_label = "End Chat" if has_chat_input else "Submit"
            with container_class:
                col1, col2 = st.columns(2)
                with col1:
                    submit_button = st.button(label=PHASE_DICT.get("button_label", default_button_label), 
                                            type="primary",
                                            key=f"submit {i}")
                with col2:
                    if PHASE_DICT.get("allow_skip", False):
                        skip_button = st.button(label="Skip Question", key=f"skip {i}")

        key = f"{PHASE_NAME}_ai_response"
        if key in st.session_state and st.session_state[key]:
            st.info(st.session_state[key], icon="🤖")

        key = f"{PHASE_NAME}_ai_score_debug"
        if key in st.session_state and st.session_state[key]:
            st.info(st.session_state[key], icon="🤖")

        key = f"{PHASE_NAME}_ai_response_revision_1"
        if key in st.session_state and st.session_state[key]:
            z = 1
            while z <= PHASE_DICT.get("max_revisions", 10):
                key = f"{PHASE_NAME}_ai_response_revision_{z}"
                if key in st.session_state and st.session_state[key]:
                    st.info(st.session_state[key], icon="🤖")
                z += 1

        key = f"{PHASE_NAME}_warning_message"
        if key in st.session_state and st.session_state[key]:
            st.warning(st.session_state[key])

        key = f"{PHASE_NAME}_error_message"
        if key in st.session_state and st.session_state[key]:
            st.error(st.session_state[key], icon="🚨")

        if submit_button:
            handle_submission(PHASE_NAME, PHASE_DICT, fields, user_input, formatted_user_prompt, selected_llm, SYSTEM_PROMPT, PHASES)
            st.rerun()

        if PHASE_DICT.get("allow_revisions", False):
            if f"{PHASE_NAME}_ai_response" in st.session_state:
                is_latest_completed_phase = i == st.session_state['CURRENT_PHASE'] or (
                        i == st.session_state['CURRENT_PHASE'] - 1 and not st.session_state.get(
                    f"{list(PHASES.keys())[i + 1]}_phase_completed", False))

                is_last_phase = (PHASE_NAME == final_phase_name)
                is_not_skipped = not st.session_state.get(f"{PHASE_NAME}_skipped", False)

                if (is_latest_completed_phase or is_last_phase) and is_not_skipped:
                    with st.expander("Revise this response?"):
                        max_revisions = PHASE_DICT.get("max_revisions", 10)
                        if f"{PHASE_NAME}_revision_count" not in st.session_state:
                            st_store(0, PHASE_NAME, "revision_count")
                        if st.session_state[f"{PHASE_NAME}_revision_count"] < max_revisions:
                            st.session_state['additional_prompt'] = st.text_input("Enter additional prompt", value="",
                                                                                  key=PHASE_NAME)
                            if st.button("Revise", key=f"revise_{i}"):
                                st.session_state[f"{PHASE_NAME}_revision_count"] += 1

                                phase_instructions = PHASE_DICT.get("phase_instructions", "")
                                user_prompt_template = PHASE_DICT.get("user_prompt", "")
                                formatted_user_prompt = format_user_prompt(user_prompt_template, user_input, PHASE_NAME,PHASES)

                                formatted_user_prompt += st.session_state['additional_prompt']

                                ai_feedback, execution_price = execute_llm_completions(SYSTEM_PROMPT,selected_llm, phase_instructions,
                                                                      formatted_user_prompt)
                                st.session_state['TOTAL_PRICE'] += execution_price

                                st_store(ai_feedback, PHASE_NAME, "ai_response_revision_" + str(
                                    st.session_state[f"{PHASE_NAME}_revision_count"]))
                                
                                # Add to chat history
                                handle_chat_history(formatted_user_prompt, ai_feedback, phase_instructions, image_urls) 
                                
                                st.rerun()
                        else:
                            st.warning("Revision limits exceeded")

        if skip_button:
            skip_phase(PHASE_NAME,PHASES,user_input)
            st.session_state[f"{PHASE_NAME}_phase_completed"] = True
            st.session_state[f"{PHASE_NAME}_skipped"] = True
            st.rerun()

        if final_key in st.session_state and i == st.session_state['CURRENT_PHASE']:
            st.success(COMPLETION_MESSAGE)
            if COMPLETION_CELEBRATION:
                celebration()

        i = min(i + 1, len(PHASES))
