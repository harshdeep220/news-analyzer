import sys
import os
from dotenv import load_dotenv

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import your graph
from agents.orchestrator import get_graph

def generate_graph_image():
    print("Generating Graph Visualization...")
    
    try:
        # Get the graph runnable
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Save to file
        with open("multi_agent_architecture.png", "wb") as f:
            f.write(graph_image)
            
        print("Success! Image saved as 'multi_agent_architecture.png'")
        
    except Exception as e:
        print(f"Error generating image: {e}")
        print("Ensure you have installed the dependencies: pip install grandalf")

if __name__ == "__main__":
    generate_graph_image()
