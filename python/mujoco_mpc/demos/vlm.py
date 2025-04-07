import os
import ast
import logging
from PIL import Image
import openai
import base64
import io
from dotenv import load_dotenv

class VLMSelector:
    def __init__(self, model_name="gpt-4o"):
        self.model = model_name
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def select_goal(self, annotated_image, labels, boxes, goal_prompt):
        object_prompts = [
            f"{i}: {label} at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            for i, (label, (x1, y1, x2, y2)) in enumerate(zip(labels, boxes))
        ]

        prompt = (
            f"You are an embodied robot. Your task is: {goal_prompt}\n"
            "Here are the detected objects:\n" +
            "\n".join(object_prompts) +
            "\nSelect the best object that matches the goal and indicate which side (e.g., front, back, left, right) you should go to."
            "\nReturn JSON: {'choice': <index>, 'position': '<relative_position>'}"
        )

        pil_image = Image.fromarray(annotated_image[:, :, :3])
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }
                ],
                max_tokens=300
            )

            text_response = response.choices[0].message.content
            dct = ast.literal_eval(text_response[text_response.rindex('{'):text_response.rindex('}')+1])
            if 'choice' in dct and 'position' in dct:
                return int(dct['choice']), dct['position'], text_response
        except Exception as e:
            logging.error("VLM (OpenAI GPT) response parsing failed: %s", e)

        return None, None, ""