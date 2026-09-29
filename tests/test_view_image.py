"""看图：view_image 工具、消息带图片、两家 provider 怎么放图、上下文管理怎么对待图片、按能力注册。

不需要 key、Docker：图片用 Pillow 现画。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from data_agent.app import build_application
from data_agent.core.context import ClearOldToolResults, CompactHistory, Context
from data_agent.core.context.base import _wire_bytes
from data_agent.core.messages import Image, LLMResponse, Message, ToolCall
from data_agent.core.tokens import estimate_image, estimate_message
from data_agent.core.tools import ToolOutput
from data_agent.llm.anthropic_provider import AnthropicProvider
from data_agent.llm.openai_provider import (
    IMAGE_OMITTED,
    TOOL_IMAGES_HEADER,
    TOOL_IMAGES_NOTE,
    OpenAICompatibleProvider,
)
from data_agent.session.codec import decode, encode
from data_agent.settings import Settings, build_provider
from data_agent.tools.paths import SandboxPaths
from data_agent.tools.view_image import MAX_EDGE, ViewImageTool

from fakes import ScriptedProvider, eval_settings, make_agent
from test_context_compaction import FakeSummarizer, add_turn, compactor

PNG = Image("image/png", "aGVsbG8=", 100, 50)


def save(path, size=(200, 100), fmt="PNG", mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new(mode, size, "white").save(path, format=fmt)
    return path


def decoded(image: Image) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(image.data)))


# ============================================================ 工具
@pytest.fixture
def tool(tmp_path):
    return ViewImageTool(SandboxPaths(tmp_path))


def test_相对路径_小图原样发(tool, tmp_path):
    path = save(tmp_path / "figures" / "fig-1.png")
    out = tool.execute({"path": "figures/fig-1.png"})
    assert not out.is_error
    assert out.content == "figures/fig-1.png（200×100）" and out.summary == "看了 figures/fig-1.png"
    [image] = out.images
    assert image.media_type == "image/png" and (image.width, image.height) == (200, 100)
    assert base64.b64decode(image.data) == path.read_bytes(), "不用缩就别重新编码"


def test_run_python报的宿主机绝对路径和容器里的路径都认(tool, tmp_path):
    path = save(tmp_path / "figures" / "a.png")
    assert not tool.execute({"path": str(path)}).is_error
    assert tool.execute({"path": "/work/figures/a.png"}).content.startswith("figures/a.png")


def test_大图缩到长边上限_告诉模型缩过(tool, tmp_path):
    save(tmp_path / "figures" / "big.png", size=(3200, 2000))
    out = tool.execute({"path": "figures/big.png"})
    [image] = out.images
    assert (image.width, image.height) == (MAX_EDGE, 980)
    assert decoded(image).size == (MAX_EDGE, 980)
    assert "3200×2000，缩小到 1568×980" in out.content


def test_JPEG还是JPEG_TIFF转成PNG(tool, tmp_path):
    save(tmp_path / "inputs" / "photo.jpg", size=(4000, 3000), fmt="JPEG")
    assert tool.execute({"path": "inputs/photo.jpg"}).images[0].media_type == "image/jpeg"
    save(tmp_path / "figures" / "forest.tiff", fmt="TIFF", mode="CMYK")
    [image] = tool.execute({"path": "figures/forest.tiff"}).images
    assert image.media_type == "image/png" and decoded(image).format == "PNG"


def test_工作目录外的文件不给看(tool, tmp_path):
    save(tmp_path.parent / "secret.png")
    for path in ("../secret.png", str(tmp_path.parent / "secret.png"), "/work/../secret.png"):
        out = tool.execute({"path": path})
        assert out.is_error and "只能读工作目录" in out.content, path


def test_找不到时列出有哪些图(tool, tmp_path):
    save(tmp_path / "figures" / "fig-1.png")
    save(tmp_path / "figures" / ".r-001.png")        # R 内核的临时文件不列
    out = tool.execute({"path": "figures/fig-2.png"})
    assert out.is_error and "figures/fig-1.png" in out.content and ".r-001" not in out.content


def test_PDF看不了_叫它另存PNG(tool, tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "forest.pdf").write_bytes(b"%PDF-1.4 ...")
    out = tool.execute({"path": "figures/forest.pdf"})
    assert out.is_error and "另存一份 PNG" in out.content


def test_能重看_旧结果可以被清理():
    assert ViewImageTool.rerunnable


def test_工具结果的图片进了历史():
    class OneImage(ViewImageTool):
        def run(self, args):
            return ToolOutput("figures/a.png（100×50）", images=(PNG,))

    agent, _ = make_agent([
        _reply(tool_calls=[ToolCall("c1", "view_image", {"path": "figures/a.png"})]),
        _reply("图没问题"),
    ], tools=[OneImage(SandboxPaths(Path(".")))])
    agent.run("看看图")
    tool_msg = agent.context.render()[2]
    assert tool_msg.role == "tool" and tool_msg.images == (PNG,)


def _reply(text="", tool_calls=()):
    return LLMResponse(text=text, tool_calls=list(tool_calls), stop_reason="tool_calls" if tool_calls else "stop")


# ============================================================ OpenAI 兼容
def history_two_images() -> list[Message]:
    return [
        Message.user("看看这两张图"),
        Message(role="assistant", tool_calls=[ToolCall("c1", "view_image", {}), ToolCall("c2", "view_image", {})]),
        Message.tool_result("c1", "figures/a.png（100×50）", images=(PNG,)),
        Message.tool_result("c2", "figures/b.png（100×50）", images=(PNG, PNG)),
        Message.assistant("好了"),
    ]


def test_openai_图片挪到这批工具结果之后的一条user消息():
    """tool 消息只能是文字；tool 消息之间插别的消息会 400，所以等这批结束再发。"""
    out = OpenAICompatibleProvider.convert_messages(history_two_images(), "sys")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user", "assistant"]
    assert out[3]["content"] == f"figures/a.png（100×50）\n\n{TOOL_IMAGES_NOTE}"
    images = out[5]["content"]
    assert images[0] == {"type": "text", "text": TOOL_IMAGES_HEADER}
    assert [p["image_url"]["url"] for p in images[1:]] == [PNG.data_url] * 3


def test_openai_对话以工具结果结尾时图片也要发出去():
    out = OpenAICompatibleProvider.convert_messages(history_two_images()[:-1], None)
    assert out[-1]["role"] == "user" and len(out[-1]["content"]) == 4


def test_openai_工具结果的图后面紧跟user_并成一条():
    """步数用完时的收尾提示紧跟在带图的工具结果后面：和那批图是同一个 user 回合，不发两条连续的 user。"""
    out = OpenAICompatibleProvider.convert_messages([*history_two_images()[:-1], Message.user("收尾")], None)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "user"]
    assert out[-1]["content"][0] == {"type": "text", "text": TOOL_IMAGES_HEADER}
    assert out[-1]["content"][-1] == {"type": "text", "text": "收尾"} and len(out[-1]["content"]) == 5


def test_openai_两条文字user照样分开():
    out = OpenAICompatibleProvider.convert_messages([Message.user("a"), Message.user("b")], None)
    assert out == [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]


def test_openai_不能看图的模型_图片换成一句说明():
    """换成 deepseek-v4-pro 接着聊，历史里还有之前的图：发过去它只会说 Unsupported Image。"""
    msgs = [*history_two_images(), Message(role="user", content="这张呢", images=(PNG,))]
    out = OpenAICompatibleProvider.convert_messages(msgs, None, vision=False)
    assert "image_url" not in json.dumps(out)
    assert out[2]["content"] == "figures/a.png（100×50）\n\n" + IMAGE_OMITTED.format(n=1)
    assert out[-1]["content"] == "这张呢\n\n" + IMAGE_OMITTED.format(n=1)


def test_openai_user消息带图():
    out = OpenAICompatibleProvider.convert_messages([Message(role="user", content="这张图", images=(PNG,))], None)
    assert out[0]["content"] == [{"type": "text", "text": "这张图"},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}]


def test_openai_没图时请求体和以前一样():
    msgs = [Message.user("q"), Message(role="assistant", tool_calls=[ToolCall("c1", "x", {})]),
            Message.tool_result("c1", "结果")]
    assert OpenAICompatibleProvider.convert_messages(msgs, None, vision=True) == \
        OpenAICompatibleProvider.convert_messages(msgs, None, vision=False)
    assert OpenAICompatibleProvider.convert_messages(msgs, None)[-1]["content"] == "结果"


# ============================================================ Anthropic
def test_anthropic_图片原生放进tool_result():
    out = AnthropicProvider.convert_messages(history_two_images())
    results = out[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"], "并行结果照样合并进一条 user 消息"
    assert results[0]["content"] == [
        {"type": "text", "text": "figures/a.png（100×50）"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
    ]
    assert len(results[1]["content"]) == 3


def test_anthropic_没图还是字符串():
    out = AnthropicProvider.convert_messages([Message.user("q")])
    assert out == [{"role": "user", "content": "q"}]
    assert AnthropicProvider.vision


# ============================================================ 上下文管理
def test_图片按面积估token():
    assert estimate_image(Image("image/png", "", 1568, 980)) == 2049
    assert estimate_image(Image("image/png", "")) == 2000, "不知道尺寸就按 2000"
    with_image = Message.tool_result("c1", "a.png", images=(PNG, PNG))
    assert estimate_message(with_image) == estimate_message(Message.tool_result("c1", "a.png")) + 2 * 7


def test_清理旧结果时图片一起换成占位():
    ctx = Context([ClearOldToolResults(trigger_tokens=0, keep_recent=0, clear_at_least=0, tools=["view_image"])])
    big = Image("image/png", "x", 1568, 980)
    ctx.add(Message.user("看图"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "view_image", {"path": "a.png"})]))
    ctx.add(Message.tool_result("c1", "figures/a.png（1568×980）", summary="看了 figures/a.png", images=(big,)))
    ctx.maintain(lambda msgs: sum(estimate_message(m) for m in msgs))
    cleared = ctx.render()[-1]
    assert cleared.images == () and "看了 figures/a.png" in cleared.content


def test_写摘要前图片换成文字占位():
    fake = FakeSummarizer()
    ctx = Context([compactor(fake)])
    ctx.add(Message.user("看图"))
    ctx.add(Message(role="assistant", tool_calls=[ToolCall("c1", "view_image", {})]))
    ctx.add(Message.tool_result("c1", "figures/a.png", images=(PNG, PNG)))
    ctx.add(Message.assistant("图没问题"))
    add_turn(ctx, 2)
    ctx.maintain(lambda msgs: 10**6)
    sent = fake.inputs[0]
    assert all(not m.images for m in sent)
    assert sent[2].content == f"figures/a.png\n{CompactHistory.IMAGE_PLACEHOLDER}\n{CompactHistory.IMAGE_PLACEHOLDER}"


def test_图片换了锚点就作废():
    """锚点的前提是前缀没变。指纹不算图片的话，图片被换掉了也认不出来。"""
    a = Message.tool_result("c1", "a.png", images=(PNG,))
    assert _wire_bytes(a) != _wire_bytes(Message.tool_result("c1", "a.png", images=(Image("image/png", "b3RoZXI="),)))
    assert _wire_bytes(a) != _wire_bytes(Message.tool_result("c1", "a.png"))


# ============================================================ 存盘
def test_会话日志里图片原样读回():
    msg = Message.tool_result("c1", "figures/a.png", summary="看了 figures/a.png", images=(PNG,))
    line = json.dumps(encode(msg), ensure_ascii=False)
    assert decode(json.loads(line)) == msg
    assert "images" not in encode(Message.user("q")), "没图不写这个字段"


# ============================================================ 按能力注册
def research_app(tmp_path, vision: bool, **settings):
    llm = ScriptedProvider()
    llm.vision = vision
    return build_application(eval_settings("research", **settings), llm=llm, work_dir=tmp_path / "work")


def test_能看图_注册view_image_提示词叫它交付前看一眼(tmp_path):
    app = research_app(tmp_path, vision=True)
    assert [t.name for t in app.tools] == ["run_python", "run_r", "read_file", "view_image"]
    prompt = app.agent.system_prompt
    assert "交付前用 `view_image` 看一眼" in prompt and "`fh_` 模板画的图不用看" in prompt
    assert app.agent.context.edits[0].tools >= {"view_image"}, "旧图可以清理"


def test_不能看图_不注册_提示词里也不提(tmp_path):
    app = research_app(tmp_path, vision=False)
    assert "view_image" not in app.tools
    assert "view_image" not in app.agent.system_prompt


def test_没有沙箱就没有图可看():
    llm = ScriptedProvider()
    llm.vision = True
    app = build_application(eval_settings("shop", python_sandbox=False), llm=llm)
    assert "view_image" not in app.tools


def test_只有Python时提示词不提fh模板(tmp_path):
    app = research_app(tmp_path, vision=True, r_sandbox=False)
    assert "view_image" in app.agent.system_prompt and "fh_" not in app.agent.system_prompt


def test_OPENAI_VISION传给provider():
    for flag in (True, False):
        llm = build_provider(Settings(provider="openai", openai_api_key="x", openai_vision=flag))
        assert llm.vision is flag
