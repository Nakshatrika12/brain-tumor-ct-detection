import os
import tempfile
import pandas as pd
import gradio as gr
import torch

from PIL import Image, ImageOps
from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import snapshot_download

# The model will be hosted separately on Hugging Face.
MODEL_ID = os.environ.get("MODEL_ID", "").strip()

if not MODEL_ID:
    raise RuntimeError(
        "MODEL_ID is not configured. Set it to your Hugging Face model repository ID."
    )

MODEL_DIR = snapshot_download(repo_id=MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict_batch(files):
    if not files:
        raise gr.Error("Please upload one or more CT images first.")

    results = []

    for file in files:
        path = file if isinstance(file, str) else file.name
        filename = os.path.basename(path)

        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                logits = model(**inputs).logits
                scores = torch.softmax(logits, dim=-1)[0].cpu()

            healthy_score = float(scores[0]) * 100
            tumor_score = float(scores[1]) * 100
            prediction = "Tumor" if tumor_score >= healthy_score else "Healthy"

            results.append({
                "Filename": filename,
                "Prediction": prediction,
                "Healthy score (%)": round(healthy_score, 2),
                "Tumor score (%)": round(tumor_score, 2),
                "Status": "Completed"
            })

        except Exception as exc:
            results.append({
                "Filename": filename,
                "Prediction": "Error",
                "Healthy score (%)": None,
                "Tumor score (%)": None,
                "Status": str(exc)[:160]
            })

    df = pd.DataFrame(results)
    completed = df[df["Status"] == "Completed"]
    tumor_count = int((completed["Prediction"] == "Tumor").sum())
    healthy_count = int((completed["Prediction"] == "Healthy").sum())
    error_count = len(df) - len(completed)

    summary = (
        f"Images submitted: {len(df)}\n"
        f"Successfully processed: {len(completed)}\n"
        f"Predicted Healthy: {healthy_count}\n"
        f"Predicted Tumor: {tumor_count}\n"
        f"Errors: {error_count}\n\n"
        "Research prototype only. These predictions are not a diagnosis "
        "and must not guide medical decisions."
    )

    csv_path = None
    if not df.empty:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as temp:
            csv_path = temp.name
            df.to_csv(temp, index=False)

    return df, summary, csv_path


with gr.Blocks(theme=gr.themes.Soft(), title="Brain CT Research Demo") as demo:
    gr.Markdown(
        """
        # Brain Tumor Detection from CT Images
        ### Vision Transformer (ViT) · Research Prototype

        Upload multiple brain CT images to view model predictions and export
        the results as a CSV report.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            image_files = gr.File(
                label="Upload CT images",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            analyze_btn = gr.Button("Analyze Images", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Textbox(
                label="Batch summary",
                lines=8,
                interactive=False
            )

    results = gr.Dataframe(
        label="Individual image predictions",
        interactive=False,
        wrap=True
    )
    csv_file = gr.File(label="Download CSV report")

    analyze_btn.click(
        fn=predict_batch,
        inputs=image_files,
        outputs=[results, summary, csv_file]
    )

    gr.Markdown(
        """
        **Important:** This is an academic research demonstration, not a
        clinically validated diagnostic system. Scores are model outputs,
        not calibrated medical probabilities. Consult qualified healthcare
        professionals for medical interpretation. Avoid uploading identifiable
        patient information.
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )


