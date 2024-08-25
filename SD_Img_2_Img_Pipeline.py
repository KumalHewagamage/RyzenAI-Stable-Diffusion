import onnxruntime as ort
import requests
from transformers import CLIPTokenizer
from optimum.onnxruntime import ORTStableDiffusionImg2ImgPipeline
from diffusers.schedulers import PNDMScheduler
from diffusers.utils import load_image
import io
import base64
import time
import matplotlib.pyplot as plt

# Paths and configurations
main_dir = "model/location"
vaip_config = "path/to/vaip_config.json"

# Load the ONNX models for VAE decoder, text encoder, and U-NET
vae_decoder_session = ort.InferenceSession(main_dir + r"\vae_decoder\model.onnx", providers=['VitisAIExecutionProvider'],
                                           provider_options=[{"config_file": vaip_config}])
text_encoder_session = ort.InferenceSession(main_dir + r"\text_encoder\model.onnx", providers=['VitisAIExecutionProvider'],
                                            provider_options=[{"config_file": vaip_config}])
unet_session = ort.InferenceSession(main_dir + r"\unet\model.onnx",
                                    providers=['VitisAIExecutionProvider'],
                                    provider_options=[{"config_file": vaip_config}])

# Load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Scheduler configuration
scheduler_config = {
    "_class_name": "PNDMScheduler",
    "_diffusers_version": "0.27.2",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None
}

# Instantiate the PNDMScheduler and ORTStableDiffusionPipeline
scheduler = PNDMScheduler(**scheduler_config)
pipeline = ORTStableDiffusionImg2ImgPipeline(
    vae_decoder_session=vae_decoder_session,
    text_encoder_session=text_encoder_session,
    unet_session=unet_session,
    tokenizer=tokenizer,
    scheduler=scheduler,
    config={},  # Add any necessary config parameters here
)


negativePrompt = "worst quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, error, sketch, duplicate, monochrome, geometry"
steps = 20


# Function to convert an image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Main function to handle API interaction and image generation
def main():
    while True:
        try:
            # Fetch the prompt from localhost:5000
            response = requests.get("http://localhost:5000")
            data = response.json()
            prompt = data.get("prompt", "")
            
            if prompt:
                # Generate the image using the pipeline
                init_image = load_image("generated_image.png")
                generated_image = pipeline(prompt,image = init_image, negative_prompt = negativePrompt,num_inference_steps = steps).images[0]
                
                # Convert image to base64 string
                img_str = image_to_base64(generated_image)

                plt.savefig("refined_generated_image.png")
                
        except Exception as e:
            print(f"Error: {e}")
        
        # Sleep for a short period to avoid busy-waiting
        time.sleep(1)

if __name__ == "__main__":
    main()
