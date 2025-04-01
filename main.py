from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os
import json
import logging
from datetime import datetime
from openai.error import OpenAIError
import time

app = FastAPI()

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thiết lập API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")
openai.api_key = openai_api_key

# Prompt hệ thống
SYSTEM_PROMPT = """
You are a smart assistant that extracts order information from Vietnamese customer messages. The message may include multiple products in a single sentence.

Your task:
- Extract: order date (use current date if missing), customer name, address, phone number, note.
- Detect and list multiple products and their quantities in format:
  "products": [{"name": "miso", "quantity": 1}, ...]
- For delivery_time: DO NOT convert to absolute time. Just extract the original text from user message, such as "sáng mai", "chiều nay", "5h chiều", etc.

Return all info as a single JSON object in the following format:

{
  "order_date": "YYYY-MM-DD",
  "customer_name": "Nguyen Van A",
  "address": "123 ABC Street, HCMC",
  "phone_number": "0909123456",
  "products": [
    {"name": "miso", "quantity": 1},
    {"name": "plain", "quantity": 2}
  ],
  "note": "Giao nhanh",
  "delivery_time": "2025-03-31 10:00"
}

If any field is missing, return it as null. Do not add extra explanations.
Return ONLY pure JSON.
"""

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    user_message = data.get("message")
    if not user_message:
        return JSONResponse(content={"status": "error", "message": "Missing 'message' field"}, status_code=400)

    logger.info("Received message: %s", user_message)

    # Tạo prompt người dùng
    prompt = f'Message: "{user_message}"'

    # Gọi GPT (retry tối đa 3 lần)
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=10
            )
            content = response["choices"][0]["message"]["content"]
            result_json = json.loads(content)
            break
        except OpenAIError as e:
            if attempt == 2:
                logger.error("OpenAI API failed after 3 attempts: %s", str(e))
                return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
            time.sleep(2)
        except json.JSONDecodeError:
            logger.error("Invalid JSON: %s", response["choices"][0]["message"]["content"])
            return JSONResponse(content={"status": "error", "message": "Invalid JSON returned from GPT"}, status_code=500)
        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    # Tạo order ID
    order_id = "ORD-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["order_id"] = order_id

    # Tạo confirmation_message tiếng Việt
    product_list = ", ".join(
        f'{p["quantity"]} {p["name"]}' for p in result_json.get("products", [])
    )
    result_json["confirmation_message"] = f"Bạn đã đặt {product_list}. Vui lòng xác nhận để bên em xử lý đơn nhé!"

    logger.info("Final GPT result: %s", result_json)
    return JSONResponse(content={"status": "ok", "result": result_json})
