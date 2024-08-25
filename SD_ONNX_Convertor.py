from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "Original\model\directory"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "Old man wearing a hat similing"
image = pipeline(prompt).images[0]
pipeline.save_pretrained("Converted\model\save\location")

import matplotlib.pyplot as plt

plt.imshow(image)
plt.axis("off")
plt.savefig("generated_image.png")
plt.show()
