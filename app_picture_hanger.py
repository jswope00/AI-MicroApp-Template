PUBLISHED = False
APP_URL = "https://voting-decision-tree.streamlit.app"

APP_TITLE = "🖼 Hang a Picture"
APP_INTRO = """🔨 This app helps you hang a painting, framed photo, or other flat artwork on the wall of a home or gallery"""

APP_HOW_IT_WORKS = """
This app helps you hang a picture by telling you where to hammer your nail based on the size of your wall and the frame.\n\nIt uses OpenAI behind the scenes to calculate this position based on the fields you complete and generates instructions accordingly. 
 """

SHARED_ASSET = {
}

HTML_BUTTON = {
    "button_text": "Learn to make an AI MicroApp like this",
    "url": "https://www.youtube.com/watch?v=jqpEJ2995IA"    
}

SYSTEM_PROMPT = """Acting as an expert in curating displays of art in homes, galleries, and museums, you will generate concise instructions in plain language understandable by a lay person on how to hang a picture, specifically where to place a nail in accordance with the measurements of the wall and picture supplied in the prompt. Your calculations will depend on the following fields from the user:

viewer_height_inches: This will be the typical standing viewer's height, and will be used to calculate the average eye level.

picture_height: This will be the vertical dimension of the picture.

drop_to_hardware: This will be the distance between the top of the picture to the wire, cleat, or other hanging hardware attached to the back of the picture. In almost every case, the hardware will be screwed or adhered to the back of the picture so as to be invisible to the viewer; the viewer will not see the nail or wire sticking out above the picture frame.

available_wall_width: This is the running length of wall space available to mount the picture; it is bounded by obstacles to the left and right such as wall corners, furniture, or other hangings.

When asked to perform a calculation, please use Python to perform the calculation and follow the provided instructions EXACTLY.
"""

PHASES = {
   "dimension_calculations": {
        "name": "Calculate your nail's position",
        "fields": {
            "info": {
                "type": "markdown",
                "body": "Use a tape measure to find the following dimensions."
            },
            "measurement_units": {
                "type": "selectbox",
                "options": ['inches', 'centimeters'],
                "label": "Are you measuring dimensions in inches or centimeters?",
                "help": "Whole numbers without fractions or decimals will suffice",
            },
            "viewer_height_inches": {
                "type": "slider",
                "min_value": 60,
                "max_value": 75,
                "value": 66,
                "label": "How tall is your average viewer in inches? (The average American height is 66 in.)",
                "showIf": {"measurement_units": "inches"},
            },
            "viewer_height_centimeters": {
                "type": "slider",
                "min_value": 155,
                "max_value": 180,
                "value": 168,
                "label": "How tall is your average viewer in centimeters? (The average American height is 168 cm.)",
                "showIf": {"measurement_units": "centimeters"},
            },
            "picture_height": {
                "type": "number_input",
                "step": 1,
                "label": "What's the height of your picture, including the frame? (You can just type a whole number)",
            },
            "drop_to_hardware": {
                "type": "number_input",
                "step": 1,
                "label": "How far below the top of the picture is the hanger on the back?",
                "help": "Measure down from the top to the place where the nail will go, whether a hook or a wire held taut, in the units you chose (inches or cm).",
            },
            "available_wall_width": {
                "type": "number_input",
                "step": 1,
                "label": "How much horizontal wall space is available?",
                "help": "Include the total span in the units you chose (inches or cm), eg between adjacent walls, nearby furniture, or pictures to the left and right.",
            },
            "picture_weight": {
                "type": "selectbox",
                "options": ['light (under 5 pounds)', 'medium (5-20 pounds)', 'heavy (over 20 pounds)'],
                "label": "How heavy is the picture?",
                "help": "Include the frame and glazing (glass or Plexi front}, if any"
            },
            "wall_type": {
                "type": "selectbox",
                "options": ['normal', 'reinforced'],
                "label": "What type of wall are you hanging on?",
                "help": "Most American rooms have normal drywall, but some galleries are reinforced with plywood backing"
            },
        },
        "user_prompt": [
            {
                "condition": {},
                "prompt": "Please use {measurement_units} in all of your output for this prompt."
            },
			{
                "condition": {},
                "prompt": """
Use Python to set [height] = ((.93 * {viewer_height_inches}) + ({picture_height}/2) - {drop_to_hardware})). Set [height] to EXACTLY the result of this calculation, with no further assumptions or calculations. Show me step by step reasoning for the calculation.
Use Python to set [distance] = ({available_wall_width}/2)
- Now tell the user to place the nail at a height of [height] off the floor and a horizontal distance of [distance] in {measurement_units} from the left obstacle. Do not tell them how to do the calculations; just do the calculations yourself and tell the user the results.'\n
                """,
            },
            {
                "condition": {"$and":[{"picture_weight": "light"},{"wall_type": "normal"}]},
                "prompt": "- Tell the user a simple nail or adhesive hook should suffice for a lightweight picture.\n",
            },
            {
                "condition": {"$and":[{"picture_weight": "medium"},{"wall_type": "normal"}]},
                "prompt": "- Recommend a picture hook or wall anchor to compensate for a medium-weight picture.  Also mention that you can place painter's tape on the wall where you plan to drill or hammer to prevent the wall from chipping and making dust. \n",
            },
            {
                "condition": {"$and":[{"picture_weight": "heavy"},{"wall_type": "normal"}]},
                "prompt": "- Recommend for a heavy picture that the user hammer one or more nails into one of the vertical wooden studs behind the wallboard, adding guidance that American homes are usually built with studs placed every 16 inches on-center.\n",
            },
            {
                "condition": {"wall_type": "reinforced"},
                "prompt": "- Explain that you can hang any reasonably sized picture by hammering one or more nails into the plywood behind the wallboard.\n",
            },
            {
                "condition": {},
                "prompt": "- After typing out the preceding information, think of a prompt that can be entered in ChatGPT to generate a diagram illustrating the measurements supplied by the user, with labeled arrows to indicate the appropriate dimensions. This schematic image should include a small nail icon or graphic positioned [height] {dimension_units} off the floor and a distance of [distance] {measurement_units} from the nearest left obstacle. The latter dimension is the distance from the nail to the nearest left obstacle; it is not the distance between the edge of the picture and the nail location. Your prompt should also draw a dashed rectangle corresponding to the picture frame, showing that the picture has a height of {picture_height} and another labeled arrow showing the picture has a hardware drop of {hardware_drop}. The lower end of the hardware drop should line up horizontally with the position of the nail. Your prompt should ask ChatGPT to draw this in the style of an architectural blueprint with white lines and text on a blue background. Your prompt should clarify that the diagram should be as easy to follow as possible, with no extraneous text or imagery. Finally, type a message to the user suggesting entering this prompt into ChatGPT.com to generate a useful diagram.\n",
            },
        ],
        "ai_response": True,
        "allow_skip": False,
        "show_prompt": True,
        "allow_revisions": True,
        #"read_only_prompt": False
    }
}

PREFERRED_LLM = "gpt-4o-mini"
LLM_CONFIG_OVERRIDE = {"gpt-4o-mini": {
        "family": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1.0,
    }
}

PAGE_CONFIG = {
    "page_title": "Hang a Picture",
    "page_icon": "️🖼",
    "layout": "centered",
    "initial_sidebar_state": "collapsed"
}

SIDEBAR_HIDDEN = True

SCORING_DEBUG_MODE = True
DISPLAY_COST = True

COMPLETION_MESSAGE = "⚠️ Chatbots can generate incorrect results; your eye is the best judge of the results. We hope this app makes it easier to hang art in your room!"
COMPLETION_CELEBRATION = False

from core_logic.main import main
if __name__ == "__main__":
    main(config=globals())
