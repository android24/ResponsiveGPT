from openai import OpenAI
from ..domain.logic import coerce_json

class JiekouChatModel:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-5.2", temperature: float = 0.4, max_tokens: int = 2048):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str) -> dict:
        text = self._complete(system, user)
        try:
            return coerce_json(text)
        except Exception:
            repair_system = system + "\n如果你刚才输出不是合法 JSON，现在必须只输出合法 JSON。"
            repair_user = (
                "上一次输出解析失败。请只返回符合 schema 的合法 JSON，不要包含任何解释、markdown 或代码块标记。\n"
                f"原始输出：\n{text}"
            )
            repaired = self._complete(repair_system, repair_user)
            return coerce_json(repaired)

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content
