import modal
import os
import sys

app = modal.App("fasthtml-test-app")

# Define Modal volumes for persistent storage
output_volume = modal.Volume.from_name("my-hackathon-outputs", create_if_missing=False)

# Create Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "python-fasthtml==0.6.9",
        "sqlite_minutils",
        "torch",
        "numpy",
        "transformers",
        "scipy",
        "scikit-learn",
        "pandas",
        "rdkit"
    )
    # Add the current directory to the image
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/"
    )
)

@app.function(
    image=image,
    volumes={"/outputs": output_volume}  # Mount the output volume
)

@modal.asgi_app()
def serve():
    # Add the root directory to Python path to ensure imports work
    sys.path.append("/")
    
    import fasthtml.common as fh
    from fastapi import Request
    from fastapi.responses import JSONResponse
    import torch
    import numpy as np
    
    # Import the prediction module
    try:
        from predict_interaction_NN_modal import BaselineModel, get_protein_features, get_ligand_features
    except ImportError as e:
        print(f"Error importing predict_interaction_NN_modal: {e}")
        print(f"Current sys.path: {sys.path}")
        print(f"Current directory contents: {os.listdir('/')}")
        raise

    app = fh.FastHTML()

    # Load the model from the Modal volume
    model_path = "/outputs/final_baseline_model.pth"  # Path in the Modal container
    model = None  # Will be loaded when needed

    def load_model():
        nonlocal model
        if model is None:
            try:
                # Determine device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")
                
                # Initialize model
                model = BaselineModel()
                
                # Load state dict with appropriate device mapping
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                
                # Move model to appropriate device
                model = model.to(device)
                model.eval()
                
                print(f"Model loaded successfully on {device}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Model path exists: {os.path.exists(model_path)}")
                raise
        return model

    @app.get("/")
    def home():
        return fh.Html(
            fh.Head(
                fh.Title("Protein-Ligand Interaction Predictor"),
                fh.Style("""
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .form-group {
                        margin-bottom: 15px;
                    }
                    .input-group {
                        margin-bottom: 10px;
                    }
                    label {
                        display: block;
                        margin-bottom: 5px;
                        font-weight: bold;
                    }
                    input[type="text"] {
                        width: 100%;
                        padding: 8px;
                        margin-bottom: 10px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    input[type="file"] {
                        width: 100%;
                        padding: 8px;
                        margin-bottom: 10px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        background-color: #f8f8f8;
                    }
                    .or-divider {
                        text-align: center;
                        margin: 10px 0;
                        color: #666;
                    }
                    button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        margin-top: 20px;
                    }
                    button:hover {
                        background-color: #45a049;
                    }
                    button:disabled {
                        background-color: #cccccc;
                        cursor: not-allowed;
                    }
                    #result {
                        margin-top: 20px;
                        padding: 15px;
                        border-radius: 4px;
                        display: none;
                    }
                    .success {
                        background-color: #dff0d8;
                        border: 1px solid #d6e9c6;
                        color: #3c763d;
                    }
                    .error {
                        background-color: #f2dede;
                        border: 1px solid #ebccd1;
                        color: #a94442;
                    }
                    #loading {
                        display: none;
                        text-align: center;
                        margin-top: 20px;
                    }
                """)
            ),
            fh.Body(
                fh.H1("Protein-Ligand Interaction Predictor"),
                fh.Form(
                    fh.Div(
                        fh.Label("Protein Sequence:"),
                        fh.Div(
                            fh.Input(type="text", name="protein_sequence", placeholder="Enter protein sequence..."),
                            class_="input-group"
                        ),
                        fh.Div(
                            fh.Span("OR", class_="or-divider")
                        ),
                        fh.Div(
                            fh.Input(type="file", name="protein_file", accept=".txt,.fasta,.fa"),
                            class_="input-group"
                        ),
                        class_="form-group"
                    ),
                    fh.Div(
                        fh.Label("Ligand SMILES:"),
                        fh.Div(
                            fh.Input(type="text", name="ligand_smiles", placeholder="Enter ligand SMILES..."),
                            class_="input-group"
                        ),
                        fh.Div(
                            fh.Span("OR", class_="or-divider")
                        ),
                        fh.Div(
                            fh.Input(type="file", name="ligand_file", accept=".txt,.smi,.smiles"),
                            class_="input-group"
                        ),
                        class_="form-group"
                    ),
                    fh.Button("Predict Interaction", type="submit", id="submit-btn"),
                    fh.Div("Processing...", id="loading"),
                    fh.Div(id="result")
                ),
                fh.Script("""
                    document.querySelector('form').addEventListener('submit', async (e) => {
                        e.preventDefault();
                        
                        const submitBtn = document.getElementById('submit-btn');
                        const loading = document.getElementById('loading');
                        const result = document.getElementById('result');
                        
                        // Disable submit button and show loading
                        submitBtn.disabled = true;
                        loading.style.display = 'block';
                        result.style.display = 'none';
                        
                        try {
                            const formData = new FormData(e.target);
                            
                            const response = await fetch('/predict', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const data = await response.json();
                            
                            result.textContent = data.message;
                            result.className = data.status === 'success' ? 'success' : 'error';
                            result.style.display = 'block';
                        } catch (error) {
                            result.textContent = 'An error occurred: ' + error.message;
                            result.className = 'error';
                            result.style.display = 'block';
                        } finally {
                            submitBtn.disabled = false;
                            loading.style.display = 'none';
                        }
                    });
                """)
            )
        )

    @app.post("/predict")
    async def predict(request: Request):
        try:
            form = await request.form()
            
            # Get protein sequence (either from text input or file)
            protein_sequence = form.get("protein_sequence", "")
            protein_file = form.get("protein_file")
            if protein_file and protein_file.filename:
                protein_sequence = await protein_file.read()
                protein_sequence = protein_sequence.decode()
            
            # Get ligand SMILES (either from text input or file)
            ligand_smiles = form.get("ligand_smiles", "")
            ligand_file = form.get("ligand_file")
            if ligand_file and ligand_file.filename:
                ligand_smiles = await ligand_file.read()
                ligand_smiles = ligand_smiles.decode()
            
            # Basic validation
            if not protein_sequence:
                return JSONResponse({
                    "status": "error",
                    "message": "Please provide a protein sequence"
                })
            
            if not ligand_smiles:
                return JSONResponse({
                    "status": "error",
                    "message": "Please provide a ligand SMILES"
                })
            
            # Load model and make prediction
            model = load_model()
            device = next(model.parameters()).device
            
            # Convert inputs to features
            protein_features = get_protein_features(protein_sequence)
            ligand_features = get_ligand_features(ligand_smiles)
            
            # Move features to the same device as the model
            protein_features = protein_features.to(device)
            ligand_features = ligand_features.to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(protein_features, ligand_features)
                probability = torch.sigmoid(prediction).item()
            
            # Format the result
            result_message = f"Prediction: {'Interaction likely' if probability > 0.5 else 'No interaction likely'} (Probability: {probability:.2%})"
            
            return JSONResponse({
                "status": "success",
                "message": result_message,
                "probability": probability
            })
            
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            })

    return app

if __name__ == "__main__":
    serve.local_entrypoint()