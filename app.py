import os
import gradio as gr
import torch
import imageio
import uuid
from PIL import Image
from unilat3d.pipelines import UniLatImageTo3DPipeline
from unilat3d.utils import render_utils, postprocessing_utils, convert
import time
import tempfile
import concurrent.futures
import threading
from datetime import datetime, timedelta
import shutil
import atexit
import numpy as np

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'  # Avoid slow benchmark on first run

# Load model
MODEL_PATH = "UniLat3D/UniLat3D"  # Change to your own model path
print(f"🔹 Loading UniLat3D model from: {MODEL_PATH}")
pipeline = UniLatImageTo3DPipeline.from_pretrained(MODEL_PATH)
pipeline.cuda()
print("✅ Model loaded successfully!")

# Temp file lifetime in seconds
TEMP_FILE_EXPIRE_SECONDS = 600  # 10 minutes

# Initialize temporary directories
def create_temp_dirs():
    base_temp = tempfile.TemporaryDirectory(prefix="3dgen_").name
    dirs = {
        "root": base_temp,
        "videos": os.path.join(base_temp, "gaussian_videos"),
        "plys": os.path.join(base_temp, "plys"),
        "splats": os.path.join(base_temp, "splat_files"),
        "glbs": os.path.join(base_temp, "glbs")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

temp_dirs = create_temp_dirs()
file_creation_times = {}

def track_file(file_path):
    """Track file creation time."""
    if file_path and os.path.exists(file_path):
        file_creation_times[file_path] = datetime.now()

def clean_expired_files():
    """Threaded task to remove expired temporary files."""
    while True:
        now = datetime.now()
        expired_files = [
            path for path, create_time in file_creation_times.items()
            if (now - create_time) > timedelta(seconds=TEMP_FILE_EXPIRE_SECONDS)
        ]
        for path in expired_files:
            try:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"🧹 Deleted expired file: {path}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"🧹 Deleted expired directory: {path}")
                del file_creation_times[path]
            except Exception as e:
                print(f"⚠️ Failed to delete {path}: {e}")
        # Check every 30 seconds
        time.sleep(30)

# Start cleanup thread
clean_thread = threading.Thread(target=clean_expired_files, daemon=True)
clean_thread.start()

def cleanup_on_exit():
    try:
        if os.path.exists(temp_dirs["root"]):
            shutil.rmtree(temp_dirs["root"])
            print(f"🧹 Cleaned up temporary directory on exit: {temp_dirs['root']}")
    except Exception as e:
        print(f"⚠️ Cleanup on exit failed: {e}")

atexit.register(cleanup_on_exit)

def gaussian_to_ply_easy(gaussian_object, output_path):
    """Save Gaussian object to PLY file using its built-in method."""
    try:
        gaussian_object.save_ply(output_path)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ PLY file saved successfully: {output_path}, Size: {file_size / 1024 / 1024:.2f} MB")
            track_file(output_path)
            return True
        else:
            print("❌ PLY file was not created successfully.")
            return False
    except Exception as e:
        print(f"⚠️ Failed to save PLY using built-in method: {e}")
        return False

def process_gaussian_video(outputs, sha256):
    """Render Gaussian video."""
    try:
        start_time = time.time()
        gs_video_path = os.path.join(temp_dirs["videos"], f"{sha256}_gaussian_color.mp4")
        video_gs = render_utils.render_video(outputs['gaussian'][0], num_frames=100, resolution=1024)
        elapsed_time = time.time() - start_time
        print(f"✅ Gaussian video rendered in {elapsed_time:.4f} seconds.")
        imageio.mimsave(gs_video_path, video_gs['color'], fps=30)
        elapsed_time = time.time() - start_time
        print(f"✅ Gaussian video saved: {gs_video_path}, Time taken: {elapsed_time:.4f} seconds.")
        track_file(gs_video_path)
        return "gaussian_video", gs_video_path
    except Exception as e:
        print(f"❌ Error generating Gaussian video: {e}")
        return "gaussian_video", None

def process_splat_viewer(outputs, sha256):
    """Generate Splat Viewer files."""
    try:
        start_time = time.time()
        ply_path = os.path.join(temp_dirs["plys"], f"{sha256}_gaussian.ply")
        gaussian_to_ply_easy(outputs['gaussian'][0], ply_path)
        
        splat_path = os.path.join(temp_dirs["splats"], f"{sha256}_unilat3d.splat")
        splat_data = convert.process_ply_to_splat(ply_path)
        convert.save_splat_file(splat_data, splat_path)
        elapsed_time = time.time() - start_time
        print(f"✅ Splat file generated: {splat_path}, Time taken: {elapsed_time:.4f} seconds.")
        track_file(splat_path)
        track_file(ply_path)
        return "splat_viewer", splat_path
    except Exception as e:
        print(f"❌ Error generating Splat Viewer: {e}")
        return "splat_viewer", None

def process_mesh_model(outputs, sha256):
    """Generate mesh model (GLB format)."""
    try:
        start_time = time.time()
        TEXTURE_SIZE = 1024
        BAKE_MODE = 'opt'
        OPT_ITER = 2500
        TARGET_VERTS = 10000
        verts = int(outputs['mesh'][0].vertices.shape[0])
        simplify_r = 0.0 if verts <= TARGET_VERTS else (1.0 - TARGET_VERTS / float(verts))
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0], outputs['mesh'][0], verbose=False,
            simplify=simplify_r, texture_size=TEXTURE_SIZE, bake_mode=BAKE_MODE, opt_iter=OPT_ITER
        )
        elapsed_time = time.time() - start_time
        print(f"✅ GLB model generated in {elapsed_time:.4f} seconds.")
        glb_path = os.path.join(temp_dirs["glbs"], f"{sha256}.glb")
        glb.export(glb_path)
        elapsed_time = time.time() - start_time
        print(f"✅ GLB model saved: {glb_path}, Time taken: {elapsed_time:.4f} seconds.")
        track_file(glb_path)
        return "mesh_model", glb_path
    except Exception as e:
        print(f"❌ Error generating mesh model: {e}")
        return "mesh_model", None

def process_image_generator(input_image, steps=55, cfg_strength=5, seed=1):
    """Generate 3D model using UniLat3D pipeline and update results asynchronously."""
    sha256 = str(uuid.uuid4())
    
    try:
        print("🔄 Running UniLat3D pipeline...")
        start_time = time.time()
        outputs = pipeline.run(
            input_image,
            seed=seed,
            unilat_sampler_params={"steps": steps, "cfg_strength": cfg_strength},
            formats=['gaussian']
        )
        elapsed_time = time.time() - start_time
        print(f"✅ Pipeline finished in {elapsed_time:.4f} seconds.")
        
        results = {"gaussian_video": None, "splat_viewer": None, "mesh_model": None}
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(process_gaussian_video, outputs, sha256): "gaussian_video",
                executor.submit(process_splat_viewer, outputs, sha256): "splat_viewer"
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_type = future_to_task[future]
                try:
                    result_type, result = future.result()
                    results[task_type] = result
                    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"🎯 Task completed: {task_type} at {formatted_time}, Elapsed: {time.time() - start_time:.2f}s")
                    
                    yield (
                        results["gaussian_video"], 
                        results["splat_viewer"], 
                        results["mesh_model"], 
                        "zoom_now" if task_type == "splat_viewer" else None
                    )
                except Exception as e:
                    print(f"❌ Task {task_type} failed: {e}")
                    results[task_type] = None
                    yield (results["gaussian_video"], results["splat_viewer"], results["mesh_model"], None)
        
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🎉 All tasks completed at {formatted_time}!")
        yield (results["gaussian_video"], results["splat_viewer"], results["mesh_model"], None)
                
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        yield None, None, None, None


# Build Gradio UI
with gr.Blocks(title="UniLat3D 3D Generator") as demo:
    zoom_trigger = gr.Textbox(value="waiting", visible=False, elem_id="zoom_trigger")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""# 🎨 UniLat3D Image to 3D Model
            Upload an image to generate the corresponding 3D model and rendering video (all files will be automatically deleted in 10 minutes)
            """)
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Image", type="pil", width=450, height=450)
                    gr.Markdown(f"""
                        ## Instructions
                        1. Upload an image
                        2. Adjust parameters (optional)
                        3. Click the "Generate 3D" button
                        4. Wait for processing to complete
                        ## Output Description
                        - **GS Splatting Viewer**: Real-time rendering result using GS splatting
                        - **Mesh Viewer**: Generated mesh result
                        - **GS Rendering Video**: Rendering result using GS
                        """)
                with gr.Column():
                    steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Sampling Steps")
                    cfg_strength = gr.Slider(minimum=1, maximum=10, value=7.5, step=0.5, label="CFG Strength")
                    seed = gr.Number(value=1, label="Seed (optional)")
                    generate_btn = gr.Button("🚀 Generate 3D", variant="primary")
                    gr.Examples(
                        examples=[
                            f'assets/images/{image}'
                            for image in os.listdir("assets/images")
                        ],
                        inputs=input_image,
                        label="Example Images (click to try!)"
                    )

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    gs_splat_output = gr.Model3D(label="3D GS splat", height=450, elem_id="3d_gs_splat")
                with gr.Column():
                    gaussian_output = gr.Video(label="GS Rendering Video", height=450)
                    mesh_output = gr.Model3D(label="3D Mesh", height=450, visible=False)

    generate_btn.click(
        fn=process_image_generator,
        inputs=[input_image, steps, cfg_strength, seed],
        outputs=[gaussian_output, gs_splat_output, mesh_output, zoom_trigger]
    )
    
    zoom_trigger.change(
        fn=None,
        inputs=[zoom_trigger],
        outputs=None,
        js="""
        function(triggerValue) {
            console.log('🔔 zoom_trigger changed:', triggerValue);
            if (triggerValue === "zoom_now") {
                const modelElement = document.getElementById('3d_gs_splat');
                const canvases = modelElement.querySelectorAll('canvas');
                if (canvases.length > 0) {
                    const event = new WheelEvent('wheel', { deltaY: -350, bubbles: true });
                    canvases[0].dispatchEvent(event);
                }
            }
        }
        """
    )

demo.launch(allowed_paths=[temp_dirs["root"]])
