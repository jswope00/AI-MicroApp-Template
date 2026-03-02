PUBLISHED = True
APP_URL = "https://rubric-generator.streamlit.app"
APP_IMAGE = "rubric_generator_flat.webp"

APP_TITLE = "Grading Rubric Generator"
APP_INTRO = """This app generates a grading rubric according to the objectives of the course and assignment. 
"""

APP_HOW_IT_WORKS = """
AI-Powered Microapps work by simplifying the creation of consistent and customized outputs from complex AI prompts. Here’s how they operate:

**1. Define the Inputs:** Begin by selecting the important user-defined variables within your AI prompt. These become the input fields in your microapp.

**2. Construct the Prompt(s):** Define the prompts with placeholders for the inputs that will come from the form. Additional internal prompts can be added, for example for scoring. Finally, prompts can be conditional so they are only sent if certain conditions are met. 

**3. User completes the form:** The microapp presents these fields as a form, and the prompt(s) is/are dynamically generated as the form is filled out. This form is designed for ease of use, allowing anyone to input the necessary details without needing to remember or interact with the AI’s underlying complexities.

**4. Deliver Consistent Results:** Once the form is completed, the microapp generates the AI’s response based on the assembled prompt. This process ensures consistent and reliable outputs, regardless of who uses the microapp. The results can then be used directly, shared, or further customized as needed.

**Benefits:**

- Ease of Use: AI-Powered Microapps make advanced AI accessible by converting complex prompts into simple forms.
- Customization: Each microapp is built to suit your specific requirements, ensuring personalized AI interactions.
- Shareability: Microapps can be easily shared across teams, organizations, or communities, providing everyone with consistent, high-quality AI outputs.
 """

SHARED_ASSET = {
}

HTML_BUTTON = {
}

SYSTEM_PROMPT = ""

PHASES = {    
    "rubric": {
        "name": "Assignment Details",
        "fields": {
            "assignment_description": {
                "type": "text_area",
                "label": "Description of the Assignment:",
                "height": 200,
                "placeholder": "e.g. In this assignment, you will create a concept map that visually represents the key concepts, processes, and relationships involved in human memory, as covered in this course. Your concept map should include at least five major concepts discussed in class (e.g., encoding, storage, retrieval, types of memory, etc.) and demonstrate how they are interconnected. You will also provide brief explanations for each connection to showcase your understanding of how these processes work together to form a cohesive memory system.",
            },
            "subject": {                
                "type": "text_input",
                "label": "Subject:",
                "placeholder": "e.g. Cognitive Psychology",
            },
            "max_points": {
                "type": "number_input",
                "label": "Maximum Points",
                "min_value": 1,
                "max_value": 100,
                "value": 100
            },
            "num_criteria": {
                "type": "slider",
                "label": "Number of Criteria",
                "min_value": 1,
                "max_value": 10,
                "value": 4,
            },
            "num_levels": {
                "type": "slider",
                "label": "Number of Achievement Levels per Criteria",
                "min_value": 2,
                "max_value": 6,
                "value": 4,
            },
            "grade_level": {
                "type": "selectbox",
                "label": "Grade Level",
                "options": ['Grade School', 'Middle School', 'University', 'Advanced Degree'],
                "index": 2
            },
            "learning_objectives": {
                "type": "text_area",
                "label": "Align with the following learning objectives:",
                "height": 200,
                "placeholder": """Demonstrate knowledge of the basic concepts and processes involved in human cognition, particularly memory.
Break down the complex interactions between different types of memory and how they contribute to overall cognitive functioning.
Use concept mapping to visually organize and relate theoretical concepts from the course to better understand their practical implications.
Present ideas clearly and concisely through a well-structured concept map, supported by logical explanations of relationships between concepts.""",

            }
        },
        "ai_response": True,
        "phase_instructions": "",
        "user_prompt": """Act as an experienced {subject} teacher and create a well crafted and clear grading rubric with {num_criteria} criteria and {num_levels} achievement levels per criteria, based on an assignment description that I will provide. Use {grade_level} student friendly language. The maximum points for the assignment is {max_points}, please make sure the rubric's maximum value adds up to that number.

            If provided, please align the rubric to the following learning objectives:\n {learning_objectives}\n\n

            Generate the rubric in the form of a table. The first row heading for the table should include the achievement levels and points. The first column on the left of the table should display the criteria. The descriptors for each component and score should be listed under the correct achievement level and points column and criteria row. Make the descriptors in the table as specific to the objectives as possible.

            Here is the assignment description for the rubric: \n {assignment_description}
            """,
        "show_prompt": True,
        "read_only_prompt": False,
        "allow_revisions": True,
        "max_revisions": 2,
    }
}

PREFERRED_LLM = "gpt-5-mini"
LLM_CONFIG_OVERRIDE = {}

SCORING_DEBUG_MODE = True
DISPLAY_COST = True

COMPLETION_MESSAGE = "You've reached the end! I hope you learned something!"
COMPLETION_CELEBRATION = False

RAG_IMPLEMENTATION = False # make true only when document exists
SOURCE_DOCUMENT = "sample.pdf" # file uploaded in source_docs if only

PAGE_CONFIG = {
    "page_title": "Rubric Generator",
    "page_icon": "️#️⃣",
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}

SIDEBAR_HIDDEN = True

from core_logic.main import main
if __name__ == "__main__":
    main(config=globals())
