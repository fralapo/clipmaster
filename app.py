import logging
from src.ui import create_ui

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("clipmaster.log", mode='w'), 
        logging.StreamHandler()
    ]
)

# --- Main Application ---
if __name__ == "__main__":
    # 1. Create the user interface and register events
    demo = create_ui()
    
    # 2. Launch the Gradio app
    demo.launch()
