from openai import OpenAI
from openai import BadRequestError
from ..domain.logic import coerce_json

class JiekouChatModel:
    """
    支持：
    1. 主模型
    2. 自动 fallback
    3. 不同模型参数兼容
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        primary_model: str = "gpt-5.2",
        fallback_model: str | None = None,
        max_completion_tokens: int = 2048,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_completion_tokens = max_completion_tokens

    def complete_json(self, system: str, user: str) -> dict:
        text = self._complete_with_fallback(system, user)

        try:
            return coerce_json(text)
        except Exception:
            repair_system = system + "\n如果你刚才输出不是合法 JSON，现在必须只输出合法 JSON。"
            repair_user = (
                "上一次输出解析失败。请只返回符合 schema 的合法 JSON，"
                "不要包含任何解释、markdown 或代码块标记。\n"
                f"原始输出：\n{text}"
            )

            repaired = self._complete_with_fallback(repair_system, repair_user)
            return coerce_json(repaired)

    def _complete_with_fallback(self, system: str, user: str) -> str:
        # 先试主模型
        try:
            return self._complete(system, user, self.primary_model)
        except BadRequestError as e:
            # 参数限制 / 某些模型路由不兼容时，自动切到 fallback
            if self.fallback_model and self.fallback_model != self.primary_model:
                print(
                    f"[WARN] primary model failed: {self.primary_model}. "
                    f"Falling back to: {self.fallback_model}. Error: {e}"
                )
                return self._complete(system, user, self.fallback_model)
            raise

    def _complete(self, system: str, user: str, model: str) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }

        # GPT-5 路由上你已经实测到 temperature/top_p 之类会报错
        # 所以这里只给非 gpt-5 模型传 temperature
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0.4

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content