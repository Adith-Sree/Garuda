#!/usr/bin/env python3
"""
Garuda Web UI — Video Upload & Processing
Uses Gradio to provide a simple drag-and-drop interface for object detection.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
import gradio as gr
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.pipeline import DetectionPipeline
from src.utils.logger import setup_logger

logger = setup_logger("garuda.app", log_file="logs/webapp.log")

def process_video(video_path, conf_threshold, frame_skip, enable_tracking):
    """
    Process the uploaded video and return the path to the result.
    """
    if video_path is None:
        return None

    logger.info("Processing uploaded video: %s", video_path)
    
    # Load default config
    config_path = "configs/model.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override settings from UI
    config["model"]["confidence_threshold"] = conf_threshold
    config["inference"]["frame_skip"] = int(frame_skip)
    config["inference"]["show_display"] = False
    config["inference"]["save_output"] = True
    config["tracking"]["enabled"] = enable_tracking
    
    # Create temporary output directory if needed
    output_dir = Path("runs/webapp")
    output_dir.mkdir(parents=True, exist_ok=True)
    config["inference"]["output_path"] = str(output_dir)

    try:
        pipeline = DetectionPipeline(config=config)
        # We need to capture the output file path. 
        # The pipeline saves to output_path/output_{timestamp}.mp4
        
        pipeline.run(source=video_path)
        
        # Find the latest output file in the output directory
        output_files = list(output_dir.glob("*.mp4"))
        if not output_files:
            logger.error("No output file generated.")
            return None
        
        latest_file = max(output_files, key=os.path.getmtime)
        logger.info("Process complete. Output saved to: %s", latest_file)
        return str(latest_file)
        
    except Exception as e:
        logger.error("Error during video processing: %s", e)
        return None

def main():
    # Define the Gradio interface
    with gr.Blocks(title="🦅 Project Garuda — UAV Detection") as demo:
        gr.Markdown("# 🦅 Project Garuda — UAV Object Detection")
        gr.Markdown("Upload a video to run real-time object detection and tracking using your custom YOLOv8 model.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.35, step=0.05, 
                        label="Confidence Threshold"
                    )
                    skip_slider = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1, 
                        label="Frame Skip (Nth frame)"
                    )
                tracking_cb = gr.Checkbox(label="Enable Tracking", value=True)
                btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Result")
        
        btn.click(
            fn=process_video,
            inputs=[video_input, conf_slider, skip_slider, tracking_cb],
            outputs=video_output
        )
        
        gr.Markdown("---")
        gr.Markdown("Built for Project Garuda — UAV Surveillance & Monitoring")

    # Launch the app
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
