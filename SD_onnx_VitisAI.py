import onnxruntime as ort
from transformers import CLIPTokenizer
from optimum.onnxruntime import ORTStableDiffusionPipeline
from diffusers.schedulers import PNDMScheduler

main_dir=r"C:\Users\K_ADMIN\Desktop\AMD GenAI\SD_FaceGen\FaceGen_ONNX"
vaip_config = "C:/Users/K_ADMIN/Desktop/AMD GenAI/GPT J/gpt-j-6B-int8-dynamic/vaip_config.json"
# Load the ONNX models for VAE decoder, text encoder, and U-NET
vae_decoder_session = ort.InferenceSession(main_dir+r"\vae_decoder\model.onnx", providers=['VitisAIExecutionProvider'],
                                    provider_options=[{"config_file":vaip_config}])
text_encoder_session = ort.InferenceSession(main_dir+r"\text_encoder\model.onnx", providers=['VitisAIExecutionProvider'],
                                    provider_options=[{"config_file":vaip_config}])
unet_session = ort.InferenceSession(main_dir+r"\unet\model.onnx", 
                                    providers=['VitisAIExecutionProvider'],
                                    provider_options=[{"config_file":vaip_config}])

# Load the tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Provided scheduler configuration dictionary
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


# Instantiate the PNDMScheduler with the provided configuration
scheduler = PNDMScheduler(**scheduler_config)

# Instantiate the ORTStableDiffusionPipeline with the scheduler
pipeline = ORTStableDiffusionPipeline(
    vae_decoder_session=vae_decoder_session,
    text_encoder_session=text_encoder_session,
    unet_session=unet_session,
    config={},  # Provide any necessary config
    tokenizer=tokenizer,
    scheduler=scheduler,  # Provide the scheduler
    # Provide additional arguments if necessary
)

# Provide the prompt
prompt = "super car on the road"

# Run inference
generated_image = pipeline(prompt).images[0]

# Display or save the generated image
# Add your code here to display or save the generated image

import matplotlib.pyplot as plt

plt.imshow(generated_image)
plt.axis("off")
#plt.savefig("generated_image.png")
plt.show()
