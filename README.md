<H2>Stable Diffusion</H2>

![alt text](Images/iVBOgKAKy6ukK2h5PW1c.png)

<H3>Converting Stable Diffusion Models to ONNX </H3>

Start with activating your RyzenAI Environment.
`conda activate <env name>`

![alt text](Images/Env_setup/image-13.png)

Install Following Dependencies.

<ul>
<li>onnxruntime</li>
<li>transformers</li>
<li>optimum</li>
<li>diffusers</li>
</ul>

```
pip install onnxruntime
pip install transformers
pip install optimum
pip install diffusers
```

Now download the your desired SD model from hugging face. I have selected runwayml/stable-diffusion-v1-5. Clone the repository with

`git clone https://huggingface.co/runwayml/stable-diffusion-v1-5`

Additionally you can let the model to be auto-downloaded during conversion process.

Use the following code to convert the model to ONNX.
```
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "<download location>/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id)
# Don't forget to save the ONNX model
save_directory = "<give a location to save the file>"
pipeline.save_pretrained(save_directory)
prompt = "super car on the road"
image = pipeline(prompt).images[0]
```
![alt text](Images/Figure_1.png)


replace `model id` line with `model_id = "runwayml/stable-diffusion-v1-5"` and it will auto download the model. If you want to use other pre-downloaded model, replace `model id` with it's folder loaction.

When Stable Diffusion models are exported to the ONNX format using Optimum, they are split into four components. They will be combined during inference session.

<ul>
<li>The text encoder</li>
<li>The VAE encoder</li>
<li>The VAE decoder</li>
<li>U-NET</li>
</ul>


Find more info in https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/models

<H3>Inference</H3>

To run the ONNX model on Vitis EP, we need to create inference sessions. 
```
class InferenceSession(
    path_or_bytes: str | bytes | PathLike,
    sess_options: Sequence | None = None,
    providers: Sequence[str] | None = None,
    provider_options: Sequence[dict[Any, Any]] | None = None,
    **kwargs: Any
)
:param path_or_bytes: Filename or serialized ONNX or ORT format model in a byte string.
:param sess_options: Session options.
:param providers: Optional sequence of providers in order of decreasing
    precedence. Values can either be provider names or tuples of (provider name, options dict). If not provided, then all available providers are used with the default precedence.
:param provider_options: Optional sequence of options dicts corresponding
    to the providers listed in 'providers'.

The model type will be inferred unless explicitly set in the SessionOptions. To explicitly set:

    so = onnxruntime.SessionOptions()
    # so.add_session_config_entry('session.load_model_format', 'ONNX') or
    so.add_session_config_entry('session.load_model_format', 'ORT')
A file extension of '.ort' will be inferred as an ORT format model. All other filenames are assumed to be ONNX format models.

'providers' can contain either names or names and options. When any options are given in 'providers', 'provider_options' should not be used.

The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider'] means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
```
Here we will use `providers=['VitisAIExecutionProvider']` as the Execution Provider and `provider_options=[{"config_file":"path to vaip_config.json"}]` vaip_config.json file as provider options.

eg:-
```
vae_decoder_session = ort.InferenceSession(main_dir+"/vae_decoder/model.onnx", providers=['VitisAIExecutionProvider'],
                                    provider_options=[{"config_file":vaip_config}])
```
We will be using `"openai/clip-vit-base-patch16"` tokenizer.

For the scheduler, we will be using following config
```
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
```

You can run the code to generate images using given prompts. Pipeline only contains basic arguments. You can add more arguments to get better results

```
pipeline = ORTStableDiffusionPipeline(
    vae_decoder_session=vae_decoder_session,
    text_encoder_session=text_encoder_session,
    unet_session=unet_session,
    config={},  
    tokenizer=tokenizer,
    scheduler=scheduler,  # Provide the scheduler
    # Provide additional arguments if necessary
)
```

Here's a list of arguments.

```
        prompt: Optional
        height: Optional[int]
        width: Optional[int]
        num_inference_steps: int 
        guidance_scale: float 
        negative_prompt: Optional
        num_images_per_prompt: 
        eta: float
        generator: Optional
        latents: Optional
        prompt_embeds: Optional
        negative_prompt_embeds: Optional
        output_type: str 
        return_dict: bool
        callback: Optional
        callback_steps: int
        guidance_rescale: float
```



