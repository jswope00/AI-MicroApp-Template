import openai
import google.generativeai as generativeai
import anthropic
import os
from dotenv import load_dotenv
import json
import pprint
import re
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.let_it_rain import rain
from config import *

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv('GOOGLE_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')

user_input = {}
function_map = {
    "text_input": st.text_input,
    "text_area": st.text_area,
    "warning": st.warning,
    "button": st.button,
    "radio": st.radio,
    "markdown": st.markdown,
    "selectbox": st.selectbox
}

def build_field(phase_name, fields):
    for field_key, field in fields.items():
        field_type = field.get("type", "")
        field_label = field.get("label", "")
        field_body = field.get("body", "")
        field_value = field.get("value", "")
        field_max_chars = field.get("max_chars", None)
        field_help = field.get("help", "")
        field_on_click = field.get("on_click", None)
        field_options = field.get("options", [])
        field_horizontal = field.get("horizontal", False)
        field_height = field.get("height", None)
        field_unsafe_html = field.get("unsafe_allow_html", False)
        field_placeholder = field.get("placeholder", "")

        kwargs = {}
        if field_label:
            kwargs['label'] = field_label
        if field_body:
            kwargs['body'] = field_body
        if field_value:
            kwargs['value'] = field_value
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
        if field_height:
            kwargs['height'] = field_height
        if field_unsafe_html:
            kwargs['unsafe_allow_html'] = field_unsafe_html
        if field_placeholder:
            kwargs['placeholder'] = field_placeholder

        key = f"{phase_name}_phase_status"

        # If the user has already answered this question:
        if key in st.session_state and st.session_state[key]:
            # Write their answer
            if f"{phase_name}_user_input_{field_key}" in st.session_state:
                if field_type != "selectbox":
                    kwargs['value'] = st.session_state[f"{phase_name}_user_input_{field_key}"]
                kwargs['disabled'] = True

        my_input_function = function_map[field_type]

        with stylable_container(
                key="large_label",
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

def call_openai_completions(system_instructions, phase_instructions, user_prompt, scoring_instructions):
    #full_system_message = f"{system_message}\n\n{scoring_instructions}"
    selected_llm = st.session_state['selected_llm']
    llm_configuration = LLM_CONFIGURATIONS[selected_llm]
    print("SELECTED LLM: " + selected_llm)
    print("LLMCONFIG: " + str(llm_configuration))
    if selected_llm in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]:
        try:
            openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = openai_client.chat.completions.create(
                model=llm_configuration["model"],
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "system", "content": phase_instructions},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=llm_configuration.get("max_tokens", 1000),
                temperature=llm_configuration.get("temperature", 1),
                top_p=llm_configuration.get("top_p", 1),
                frequency_penalty=llm_configuration.get("frequency_penalty", 0),
                presence_penalty=llm_configuration.get("presence_penalty", 0)
            )
            input_price = int(response.usage.prompt_tokens) * llm_configuration["price_input_token_1M"] / 1000000
            output_price = int(response.usage.completion_tokens) * llm_configuration["price_output_token_1M"] / 1000000
            total_price = input_price + output_price
            st.session_state['TOTAL_PRICE'] += total_price
            return response.choices[0].message.content
        except Exception as e:
            st.write(f"**OpenAI Error Response:** {selected_llm}")
            st.error(f"Error: {e}")
    if selected_llm in ["gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"]:
        try:
            generativeai.configure(api_key=gemini_api_key)
            generation_config = {
              "temperature": llm_configuration["temperature"],
              "top_p": llm_configuration.get("top_p", 1),
              "max_output_tokens": llm_configuration.get("max_tokens", 1000),
              "response_mime_type": "text/plain",
            }
            model = generativeai.GenerativeModel(
                llm_configuration["model"],
                generation_config=generation_config,
                system_instruction=system_instructions
                )
            chat_session = model.start_chat(
                history=[
                {
                  "role": "model",
                  "parts": [
                    phase_instructions,
                  ],
                }
              ]
            )
            gemini_response = chat_session.send_message(user_prompt)
            gemini_response_text = gemini_response.text

            return gemini_response_text
        except Exception as e:
            st.write("**Gemini Error Response:**")
            st.error(f"Error: {e}")
            print(f"Error: {e}")
    if selected_llm in ["claude-opus", "claude-sonnet", "claude-haiku"]:
        try:
            client = anthropic.Anthropic(api_key=claude_api_key)
            anthropic_response = client.messages.create(
                model=llm_configuration["model"],
                max_tokens=llm_configuration["max_tokens"],
                temperature=llm_configuration["temperature"],
                system=system_instructions,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Hello. I will give you some instructions via the assistant prompt next. "}]},
                    {"role": "assistant", "content": [{"type": "text", "text": phase_instructions}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                ]
            )
            input_price = int(anthropic_response.usage.input_tokens) * llm_configuration[
                "price_input_token_1M"] / 1000000
            output_price = int(anthropic_response.usage.output_tokens) * llm_configuration[
                "price_output_token_1M"] / 1000000
            total_price = input_price + output_price
            response_cleaned = '\n'.join([block.text for block in anthropic_response.content if block.type == 'text'])
            st.session_state['TOTAL_PRICE'] += total_price
            return response_cleaned
        except Exception as e:
            st.write(f"**Anthropic Error Response: {selected_llm}**")
            st.error(f"Error: {e}")
            print(f"Error: {e}")

def st_store(input, phase_name, phase_key, field_key = ""):
    if field_key:
        key = f"{phase_name}_{field_key}_{phase_key}"
    else:
        key = f"{phase_name}_{phase_key}"
    st.session_state[key] = input

def build_scoring_instructions(rubric):
    scoring_instructions = f"""
    Please score the user's previous response based on the following rubric: \n{rubric}
    \n\nPlease output your response as JSON, using this format: {{ "[criteria 1]": "[score 1]", "[criteria 2]": "[score 2]", "total": "[total score]" }}
    """
    return scoring_instructions

def extract_score(text):
    pattern = r'"total":\s*"?(\d+)"?'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    else:
        return 0

def check_score(PHASE_NAME):
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

def skip_phase(PHASE_NAME, No_Submit=False):
    phase_fields = PHASES[PHASE_NAME]["fields"]
    for field_key in phase_fields:
        st_store(user_input, PHASE_NAME, "user_input", field_key)
        if No_Submit == False:
            st.session_state[f"{PHASE_NAME}_ai_response"] = "This phase was skipped."
    st.session_state[f"{PHASE_NAME}_phase_status"] = True
    st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)

def celebration():
    rain(
        emoji="🥳",
        font_size=54,
        falling_speed=5,
        animation_length=1,
    )


def main():

    if 'TOTAL_PRICE' not in st.session_state:
        st.session_state['TOTAL_PRICE'] = 0

    with st.sidebar:
        selected_llm = st.selectbox("Select Language Model", options=LLM_CONFIGURATIONS.keys(), key="selected_llm")
        st.write("Price : ${:.6f}".format(st.session_state['TOTAL_PRICE']))

    

    if 'CURRENT_PHASE' not in st.session_state:
        st.session_state['CURRENT_PHASE'] = 0
    #if 'results' not in st.session_state:
    #    st.session_state['results'] = {}

    st.title(APP_TITLE)
    st.markdown(APP_INTRO)

    if APP_HOW_IT_WORKS:
        with st.expander("Learn how this works", expanded=False):
            st.markdown(APP_HOW_IT_WORKS)

    if SHARED_ASSET:
        # Download button for the PDF
        with open(SHARED_ASSET["path"], "rb") as asset_file:
            st.download_button(label=SHARED_ASSET["button_text"],
                               data=asset_file,
                               file_name=SHARED_ASSET["name"],
                               mime="application/octet-stream")

    if HTML_BUTTON:
        st.link_button(label=HTML_BUTTON["button_text"], url=HTML_BUTTON["url"])

    i = 0

    while i <= st.session_state['CURRENT_PHASE']:
        submit_button = False

        skip_button = False
        final_phase_name = list(PHASES.keys())[-1]
        final_key = f"{final_phase_name}_ai_response"

        PHASE_NAME = list(PHASES.keys())[i]
        PHASE_DICT = PHASES[PHASE_NAME]
        fields = PHASE_DICT["fields"]

        st.write(f"#### Phase {i+1}: {PHASE_DICT['name']}")


        build_field(PHASE_NAME, fields)

        key = f"{PHASE_NAME}_phase_status"

        # Check phase status to automatically continue if it's a no_submission phase
        if PHASE_DICT.get("no_submission", False) == True:
            if key not in st.session_state:
                st.session_state[key] = True
                st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)

        if key not in st.session_state:
            st.session_state[key] = False
        # If the phase isn't passed and it isn't a recap of the final phase, then give the user a submit button
        if st.session_state[key] != True and final_key not in st.session_state:
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    submit_button = st.button(label=PHASE_DICT.get("button_label", "Submit"), type="primary",
                                              key="submit " + str(i))
                with col2:
                    if PHASE_DICT.get("allow_skip", False):
                        skip_button = st.button(label="Skip Question", key="skip " + str(i))


        key = f"{PHASE_NAME}_ai_response"
        # If the AI has responded:
        if key in st.session_state:
            # Then print the stored AI Response
            st.info(st.session_state[key], icon="🤖")
        key = f"{PHASE_NAME}_ai_result"
        # If we are showing a score:
        if key in st.session_state and SCORING_DEBUG_MODE == True:
            # Then print the stored AI Response
            st.info(st.session_state[key], icon="🤖")

        if submit_button:
            #Store all the user date
            for field_key, field in fields.items():
                st_store(user_input[field_key], PHASE_NAME, "user_input", field_key)
                user_response = user_input[field_key]
            
            #If there is no hard-coded response, then submit to AI
            if PHASE_DICT.get("ai_response", "") == "":
                system_instructions = SYSTEM_INSTRUCTIONS
                phase_instructions = PHASE_DICT.get("phase_instructions", "")
                user_prompt = PHASE_DICT.get("user_prompt","")
                scoring_instructions = ""
                ai_feedback = call_openai_completions(system_instructions, phase_instructions, user_prompt, scoring_instructions)
                st_store(ai_feedback, PHASE_NAME, "ai_response")
                st.info(ai_feedback)
            #If AI Response is hard-coded, then use that
            else:
                res_box = st.info(body="", icon="🤖")
                result = ""
                report = []
                
                hard_coded_message = PHASE_DICT['ai_response']
                #TO-DO: This is supposed to stream, but it does not right now.
                for char in hard_coded_message:
                    result += char
                    report.append(char)
                    res_box.info(body=f'{result}', icon="🤖")
                st.session_state[f"{PHASE_NAME}_ai_response"] = hard_coded_message

            if PHASE_DICT.get("scored_phase", "") == True:
                if "rubric" in PHASE_DICT:
                    scoring_instructions = build_scoring_instructions(PHASE_DICT["rubric"])
                    ai_feedback = call_openai_completions(system_instructions, phase_instructions, user_prompt, scoring_instructions)

                    #TEMP!!!:
                    st_store(100, PHASE_NAME, "ai_score")

                    if check_score(PHASE_NAME):
                        st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
                        # Rerun Streamlit to refresh the page
                        st.rerun()
                    else:
                        st.warning("You haven't passed. Please try again.")
                else:
                    st.error('You need to include a rubric for a scored phase', icon="🚨")
            else:
                st.session_state[f"{PHASE_NAME}_phase_status"] = True
                st.session_state['CURRENT_PHASE'] = min(st.session_state['CURRENT_PHASE'] + 1, len(PHASES) - 1)
                # Rerun Streamlit to refresh the page
                st.rerun()

        if skip_button:
            skip_phase(PHASE_NAME)
            st.rerun()

        if final_key in st.session_state and i == st.session_state['CURRENT_PHASE']:
            st.success(COMPLETION_MESSAGE)
            if COMPLETION_CELEBRATION:
                celebration()

        i = min(i + 1, len(PHASES))

if __name__ == "__main__":
    main()
