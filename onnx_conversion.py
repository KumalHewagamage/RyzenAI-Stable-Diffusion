from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = r"C:\Users\K_ADMIN\Desktop\AMD GenAI\HF Stuff\FaceGen"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "Old man wearing a hat similing"
image = pipeline(prompt).images[0]
pipeline.save_pretrained(r"C:\Users\K_ADMIN\Desktop\AMD GenAI\SD_FaceGen\FaceGen_ONNX")

import matplotlib.pyplot as plt

plt.imshow(image)
plt.axis("off")
plt.savefig("generated_image.png")
plt.show()
