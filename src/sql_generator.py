import time
import re
import threading
from src.config import (
    BASE_MODEL_NAME,
    LORA_MODEL_NAME,
    SQL_MAX_NEW_TOKENS,
    SQL_TEMPERATURE
)

class SQLGenerator:
    def __init__(self, base_model_name=BASE_MODEL_NAME, lora_model_name=LORA_MODEL_NAME, auto_load=True):
        self.base_model_name = base_model_name
        self.lora_model_name = lora_model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.status = "not_started"  # not_started, loading, ready, error
        self.load_error = None
        self._lock = threading.Lock()

        if auto_load:
            self.load_in_background()

    def load_in_background(self):
        """Memuat model di thread background agar server Flask dapat langsung start tanpa terblokir."""
        thread = threading.Thread(target=self._load_model, daemon=True, name="sql-model-loader")
        thread.start()

    def _load_model(self):
        with self._lock:
            self.status = "loading"
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[INFO] Inisialisasi Text-to-SQL Model pada device: {self.device}")

                print(f"[INFO] Memuat Tokenizer dari: {self.lora_model_name}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_name)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

                print(f"[INFO] Memuat Base Model: {self.base_model_name} (membutuhkan waktu jika mengunduh/membaca bobot)...")
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                device_map = "auto" if self.device == "cuda" else None

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map
                )

                print(f"[INFO] Memuat LoRA Adapter: {self.lora_model_name}...")
                self.model = PeftModel.from_pretrained(base_model, self.lora_model_name)
                self.model.eval()
                self.status = "ready"
                self.load_error = None
                print("[INFO] Model Text-to-SQL berhasil dimuat dan siap melayani request.")
            except Exception as e:
                self.status = "error"
                self.load_error = str(e)
                print(f"[WARNING] Model Text-to-SQL belum dapat dimuat: {e}")

    def is_ready(self) -> bool:
        return self.status == "ready" and self.model is not None and self.tokenizer is not None

    def clean_sql(self, raw_text: str) -> str:
        """Membersihkan markdown code blocks dan whitespace pada output SQL."""
        text = raw_text.strip()
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if sql_match:
            text = sql_match.group(1).strip()
        text = text.replace("```", "").strip()
        return text

    def generate(self, schema: str, question: str) -> dict:
        """
        Menghasilkan kueri SQL berdasarkan database schema dan pertanyaan bahasa Indonesia.
        """
        if not self.is_ready():
            msg = "Model Text-to-SQL sedang dalam proses inisialisasi bobot (loading)." if self.status == "loading" else f"Model gagal dimuat: {self.load_error}"
            raise RuntimeError(msg)

        import torch

        start_time = time.time()

        prompt_content = f"Database Schema:\n{schema.strip()}\n\nQuestion:\n{question.strip()}"
        messages = [
            {
                "role": "system",
                "content": "Anda adalah asisten AI yang ahli dalam mengonversi pertanyaan bahasa Indonesia ke SQL berdasarkan skema database yang diberikan."
            },
            {"role": "user", "content": prompt_content}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_text = f"<|im_start|>system\nAnda adalah asisten AI yang ahli dalam mengonversi pertanyaan bahasa Indonesia ke SQL berdasarkan skema database yang diberikan.<|im_end|>\n<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"

        target_device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = self.tokenizer([formatted_text], return_tensors="pt").to(target_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=SQL_MAX_NEW_TOKENS,
                do_sample=SQL_TEMPERATURE > 0,
                temperature=SQL_TEMPERATURE if SQL_TEMPERATURE > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_token_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        cleaned_sql = self.clean_sql(raw_output)

        duration_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "sql": cleaned_sql,
            "raw_output": raw_output,
            "execution_time_ms": duration_ms
        }

# Instance singleton (otomatis mulai background loading saat server boot)
sql_generator = SQLGenerator()
