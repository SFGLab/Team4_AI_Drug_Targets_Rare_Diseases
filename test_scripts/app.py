import modal
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from typing import Optional
import tempfile
import shutil

# Import our existing training and prediction functions (will be available in Modal container)
# Note: These imports rely on the local directory being added to the image
from train_NN_modal import modal_main as train_main
from predict_interaction_NN_modal import predict_on_modal as predict_main

# Create Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "python-multipart", # Required for file uploads with FastAPI
        "uvicorn", # Required for running the FastAPI app
        "aiofiles" # Required for async file operations
    )
    # Add the current directory to the image. This makes local code (like train_NN_modal.py) importable.
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/"
    )
)

# Define Modal volumes for persistent storage
# Assuming input data and the trained model are in my-hackathon-data and my-hackathon-outputs respectively
volume = modal.Volume.from_name("my-hackathon-data", create_if_missing=False)
output_volume = modal.Volume.from_name("my-hackathon-outputs", create_if_missing=True)

# Create Modal app instance
app = modal.App("ligand-protein-webapp")

# Create the static directory and HTML file if they don't exist
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Create the HTML file for the web interface
with open(os.path.join(static_dir, "index.html"), "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Ligand-Protein Interaction Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input[type="file"], input[type="text"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
        }
        .success {
            background-color: #dff0d8;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            color: #a94442;
        }
    </style>
</head>
<body>
    <h1>Ligand-Protein Interaction Web App</h1>
    
    <div class="section">
        <h2>Upload Files</h2>
        <div class="form-group">
            <label for="file">Select File:</label>
            <input type="file" id="file" name="file">
            <button onclick="uploadFile()">Upload</button>
        </div>
    </div>

    <div class="section">
        <h2>Train Model</h2>
        <div class="form-group">
            <label for="trainDataset">Dataset File:</label>
            <input type="text" id="trainDataset" placeholder="dataset.csv">
            <label for="trainProteinEmbeddings">Protein Embeddings:</label>
            <input type="text" id="trainProteinEmbeddings" placeholder="protein_embeddings.npy">
            <label for="trainLigandEmbeddings">Ligand Embeddings:</label>
            <input type="text" id="trainLigandEmbeddings" placeholder="ligand_embeddings.npy">
            <button onclick="trainModel()">Train</button>
        </div>
    </div>

    <div class="section">
        <h2>Make Predictions</h2>
        <div class="form-group">
            <label for="predictInput">Input File:</label>
            <input type="text" id="predictInput" placeholder="input.csv">
            <label for="predictModel">Model File:</label>
            <input type="text" id="predictModel" placeholder="model.pth">
            <label for="predictProteinEmbeddings">Protein Embeddings:</label>
            <input type="text" id="predictProteinEmbeddings" placeholder="protein_embeddings.npy">
            <label for="predictLigandEmbeddings">Ligand Embeddings:</label>
            <input type="text" id="predictLigandEmbeddings" placeholder="ligand_embeddings.npy">
            <button onclick="makePredictions()">Predict</button>
        </div>
    </div>

    <div class="section">
        <h2>Available Files</h2>
        <button onclick="listFiles()">Refresh File List</button>
        <div id="fileList"></div>
    </div>

    <div id="status"></div>

    <script>
        function showStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = isError ? 'error' : 'success';
        }

        async function uploadFile() {
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];
            if (!file) {
                showStatus('Please select a file to upload', true);
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                showStatus(result.status === 'success' ? 
                    'File uploaded successfully' : 
                    'Error uploading file: ' + result.message,
                    result.status !== 'success'
                );
                listFiles();
            } catch (error) {
                showStatus('Error uploading file: ' + error, true);
            }
        }

        async function trainModel() {
            const dataset = document.getElementById('trainDataset').value;
            const proteinEmbeddings = document.getElementById('trainProteinEmbeddings').value;
            const ligandEmbeddings = document.getElementById('trainLigandEmbeddings').value;

            if (!dataset || !proteinEmbeddings || !ligandEmbeddings) {
                showStatus('Please fill in all fields', true);
                return;
            }

            const formData = new FormData();
            formData.append('dataset_file', dataset);
            formData.append('protein_embeddings', proteinEmbeddings);
            formData.append('ligand_embeddings', ligandEmbeddings);

            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                showStatus(result.status === 'success' ? 
                    'Training started successfully' : 
                    'Error starting training: ' + result.message,
                    result.status !== 'success'
                );
            } catch (error) {
                showStatus('Error starting training: ' + error, true);
            }
        }

        async function makePredictions() {
            const inputFile = document.getElementById('predictInput').value;
            const modelFile = document.getElementById('predictModel').value;
            const proteinEmbeddings = document.getElementById('predictProteinEmbeddings').value;
            const ligandEmbeddings = document.getElementById('predictLigandEmbeddings').value;

            if (!inputFile || !modelFile || !proteinEmbeddings || !ligandEmbeddings) {
                showStatus('Please fill in all fields', true);
                return;
            }

            const formData = new FormData();
            formData.append('input_file', inputFile);
            formData.append('model_file', modelFile);
            formData.append('protein_embeddings', proteinEmbeddings);
            formData.append('ligand_embeddings', ligandEmbeddings);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showStatus('Predictions completed successfully');
                    window.location.href = `/download/${result.output_file}`;
                } else {
                    showStatus('Error making predictions: ' + result.message, true);
                }
            } catch (error) {
                showStatus('Error making predictions: ' + error, true);
            }
        }

        async function listFiles() {
            try {
                const response = await fetch('/files');
                const result = await response.json();
                if (result.status === 'success') {
                    const fileList = document.getElementById('fileList');
                    fileList.innerHTML = `
                        <h3>Data Files:</h3>
                        <ul>${result.data_files.map(f => `<li>${f}</li>`).join('')}</ul>
                        <h3>Output Files:</h3>
                        <ul>${result.output_files.map(f => `<li>${f}</li>`).join('')}</ul>
                    `;
                } else {
                    showStatus('Error listing files: ' + result.message, true);
                }
            } catch (error) {
                showStatus('Error listing files: ' + error, true);
            }
        }

        // List files when page loads
        listFiles();
    </script>
</body>
</html>
    """)

# Create FastAPI app instance
web_app = FastAPI(title="Ligand-Protein Interaction Web App")

# Add CORS middleware to allow cross-origin requests from the web interface
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - Serve the entire static directory from the root path
# This will automatically serve index.html when accessing the root URL
# The directory is assumed to be in the same location relative to app.py inside the container as it is locally
web_app.mount("/", StaticFiles(directory="/static", html=True), name="static")

# Define Modal function to handle training requests
@app.function(image=image, volumes={"/data": volume, "/outputs": output_volume})
def train_model_modal(
    dataset_path: str,
    protein_embeddings_path: str,
    ligand_embeddings_path: str
):
    """Train the model using Modal"""
    # Call the original training function
    try:
        # Ensure the original script can find its imported modules by setting up the path
        import sys
        sys.path.append("/") # Add the root of the container (where our code is mounted) to the Python path
        
        auc = train_main.remote(
            dataset_path=dataset_path,
            protein_embeddings_path=protein_embeddings_path,
            ligand_embeddings_path=ligand_embeddings_path
        )
        return {"status": "success", "auc": auc}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Define Modal function to handle prediction requests
@app.function(image=image, volumes={"/data": volume, "/outputs": output_volume})
def make_predictions_modal(
    input_csv_path: str,
    model_path: str,
    output_csv_path: str,
    protein_embeddings_path: str,
    ligand_embeddings_path: str
):
    """Make predictions using Modal"""
    # Call the original prediction function
    try:
        # Ensure the original script can find its imported modules by setting up the path
        import sys
        sys.path.append("/") # Add the root of the container (where our code is mounted) to the Python path
        
        predict_main.remote(
            input_csv_path=input_csv_path,
            model_path=model_path, # Assuming model is saved in outputs volume
            output_csv_path=output_csv_path,
            protein_embeddings_path=protein_embeddings_path,
            ligand_embeddings_path=ligand_embeddings_path
        )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to handle file uploads
@web_app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads and save to Modal volume"""
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            # Read the uploaded file in chunks and write to the temporary file
            # Use aiofiles for async file operations
            import aiofiles
            async with aiofiles.open(tmp.name, 'wb') as out_file:
                while content := await file.read(1024):
                    await out_file.write(content)
            tmp_path = tmp.name

        # Copy the file from the temporary location to the Modal volume path
        # This needs to be done in a blocking way as shutil does not support async
        # In a real-world app, consider using a Modal function or a separate process for large files
        volume_path = os.path.join("/data", file.filename)
        shutil.copyfile(tmp_path, volume_path)

        # Clean up the temporary file
        os.unlink(tmp_path)

        return {"status": "success", "filename": file.filename, "volume_path": volume_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to trigger model training on Modal
@web_app.post("/train")
async def train(
    dataset_file: str = Form(...),
    protein_embeddings: str = Form(...),
    ligand_embeddings: str = Form(...)
):
    """Trigger model training on Modal"""
    try:
        # Trigger the Modal training function
        result = train_model_modal.remote(
            dataset_path=os.path.join("/data", dataset_file),
            protein_embeddings_path=os.path.join("/data", protein_embeddings),
            ligand_embeddings_path=os.path.join("/data", ligand_embeddings)
        )
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to trigger predictions on Modal
@web_app.post("/predict")
async def predict(
    input_file: str = Form(...),
    model_file: str = Form(...),
    protein_embeddings: str = Form(...),
    ligand_embeddings: str = Form(...)
):
    """Trigger predictions on Modal""" # Added docstring for clarity
    try:
        output_filename = f"predictions_{input_file}"
        output_volume_path = os.path.join("/outputs", output_filename)
        
        # Trigger the Modal prediction function
        result = make_predictions_modal.remote(
            input_csv_path=os.path.join("/data", input_file),
            model_path=os.path.join("/outputs", model_file), # Assuming model is saved in outputs volume
            output_csv_path=output_volume_path,
            protein_embeddings_path=os.path.join("/data", protein_embeddings),
            ligand_embeddings_path=os.path.join("/data", ligand_embeddings)
        )
        # The make_predictions_modal function handles saving the output and metrics to the volume
        
        # Return the filename where the predictions are saved
        return {"status": "success", "output_file": output_filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to list files in volumes
@web_app.get("/files")
async def list_files():
    """List available files in the volumes"""
    try:
        # List files in the data and output volumes
        data_files = os.listdir("/data")
        output_files = os.listdir("/outputs")
        return {
            "status": "success",
            "data_files": data_files,
            "output_files": output_files
        }
    except FileNotFoundError as e:
        # Handle case where volumes might not be initialized with any files yet
        print(f"Error listing files: {e}. Volumes might be empty.")
        return {"status": "success", "data_files": [], "output_files": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# FastAPI endpoint to download a file from the output volume
@web_app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file from the output volume"""
    try:
        file_path = os.path.join("/outputs", filename)
        # Check if the file exists in the output volume
        if os.path.exists(file_path):
            # Use FileResponse to serve the file for download
            return FileResponse(file_path, filename=filename)
        else:
            # Return a 404 error if the file is not found
            return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Modal entrypoint to serve the FastAPI app
@app.function(image=image, volumes={"/data": volume, "/outputs": output_volume})
@modal.asgi_app()
def entrypoint():
    """Modal entrypoint to serve the FastAPI app"""
    return web_app 