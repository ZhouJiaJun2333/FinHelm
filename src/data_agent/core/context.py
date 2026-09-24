"""上下文管理 —— 「对话变长之后怎么办」的扩展点。

现在只有最朴素的实现。先把接口定下来，以后加摘要压缩 / 向量检索 / 记忆注入
只要写一个新的子类，agent.py 一行不用改。

⚠️ 新手最容易踩的坑：
    不能简单地「只保留最后 N 条消息」。assistant 的 tool_calls 和后面 role="tool"
    的结果**必须成对出现**，从中间一刀切下去，API 直接返回 400。
    真要裁剪，得以「一个完整回合」为单位 —— 见 TurnWindowContext。

「半截状态毒化历史」一共有三个入口，但后果轻重不一样：

    1. assistant 有 tool_calls 但没有对应的 tool 结果
       → 两家都 400，而且是**永久**的：坏消息一直躺在历史里，之后每一轮
         都报同样的错，一次失败升级成整个会话报废。
    2. 连着两条 user 消息（提问后这轮失败了，用户又问一次）
       → **不报错**。Anthropic 文档原话是连续同角色的消息「会被合并成一条」，
         OpenAI 兼容接口也照收。坏在语义：失败那轮的问题和新问题被悄悄拼在
         一起，模型以为用户一口气问了两遍 / 两件事。不报错反而更难发现。
    3. 历史以 tool 结果或 nudge 的 user 消息结尾，没有收尾的 assistant
       → 下一轮追加 user 后同样被合并，不报错；但模型不知道上一轮卡住了。

前两个走异常路径，靠 Agent.run() 的事务语义（snapshot/restore）兜住；
第三个走**正常返回**路径（步数耗尽），事务照常提交，所以由 run() 自己
补一条收尾的 assistant 消息。

⚠️ 反过来，「以 assistant 结尾」发请求是会报错的：Anthropic 把末尾的 assistant
   当成 prefill（让模型接着这段往下写），而 Opus 4.6 之后的模型不再支持
   prefill，直接 400。所以历史收尾是 assistant 没问题，但**发请求之前**
   末尾必须是 user / tool 结果 —— 这就是 finish_turn 继续时要补 nudge 的原因。

tests/ 里的哨兵仍然按「严格交替 + tool_use 配对」来验。交替不是 API 硬约束，
是我们自己要的不变量：它保证历史里每个问题都有且只有一个回答。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable, Iterable

from .events import ContextCleared
from .messages import Message
from .tokens import estimate_message, estimate_text

# 给一份消息列表估 token 数。由 Agent 提供 —— 只有它知道系统提示词和工具定义。
Measure = Callable[[list[Message]], int]


class BaseContext(ABC):
    """上下文管理器接口。"""

    @abstractmethod
    def add(self, message: Message) -> None:
        """追加一条消息。"""

    @abstractmethod
    def render(self) -> list[Message]:
        """返回这一轮真正要发给模型的消息列表。

        注意方法名是 render 不是 messages —— 它强调这是「算出来的视图」，
        可以和内部存的历史不一样（压缩、裁剪、插入检索结果都发生在这里）。
        """

    def maintain(self, measure: Measure) -> ContextCleared | None:
        """每次请求模型**之前**调用一次，给上下文一个整理自己的机会。

        默认什么都不做。需要压缩的子类覆盖它，做了事就返回一个事件供界面展示。

        为什么是「请求前」而不是「追加消息时」：要不要压缩取决于**整个请求**
        有多大（含系统提示词、工具定义），这只有在马上要发的时候才算得准。
        """
        return None

    @abstractmethod
    def clear(self) -> None:
        """清空。"""

    @abstractmethod
    def snapshot(self) -> Any:
        """拍一张当前状态的快照，交给 restore() 用。

        用来给 Agent.run() 提供**事务语义**：一轮对话要么完整完成，
        要么历史回到进来之前的样子，不留任何残骸。

        为什么需要：一轮失败（被截断、网络错、用户 Ctrl-C）时，历史里可能
        留下「有提问没回答」或「有工具调用没结果」的半截状态。后者会让
        **之后每一轮**请求都 400；前者不报错，但会和下一个问题被合并成一条。
        """

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """回滚到 snapshot() 拍下的状态。"""


class FullContext(BaseContext):
    """全量保留。够用到你开始撞上下文上限为止。"""

    def __init__(self) -> None:
        self._history: list[Message] = []

    def add(self, message: Message) -> None:
        self._history.append(message)

    def render(self) -> list[Message]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    # 快照存整个列表的浅拷贝，而不是只记长度。
    # 只记长度对「只追加」的实现够用，但压缩类上下文会**改写**已有消息，
    # 那时候长度回滚不了内容。拷贝一份最省心，历史规模也就几百条。
    def snapshot(self) -> list[Message]:
        return list(self._history)

    def restore(self, snapshot: list[Message]) -> None:
        self._history[:] = snapshot

    def __len__(self) -> int:
        return len(self._history)


class ToolResultClearingContext(FullContext):
    """超过阈值时，把较早的工具结果换成一句占位文字。最便宜的一种压缩。

    参数照抄 Anthropic 服务端的同款功能（context editing 的 clear_tool_uses）：

        trigger_tokens  请求估算超过它才动手
        keep_recent     最近几条工具结果不动（模型多半正在用）
        clear_at_least  一次至少要省下这么多，否则不清 —— 见下面「缓存」
        exclude_tools   这些工具的结果永远不清

    为什么先清工具结果：它是上下文里最大的一块（SQL 结果表格），而且
    **能重新拿到** —— 调用参数（那条 SQL）还留在 assistant 消息里，模型需要
    数据时再跑一次就行。用户的原话、模型的结论清掉就找不回来了，那是摘要
    （第 4 步）的事。

    ── 原件不删，只改视图 ────────────────────────────────────────────
    内部历史一个字不动，清理只记在 _cleared 里，render() 时才换成占位。
    好处：原件还在（以后能做「按需取回」），回滚只要连 _cleared 一起恢复。

    ── 缓存：为什么要攒一批才清 ──────────────────────────────────────
    prompt 缓存是前缀匹配。改了第 k 条消息，第 k 条之后的缓存全部失效，
    下一次请求要重新写缓存（Anthropic 写缓存比正常输入还贵 25%）。
    如果规则是「永远只留最近 3 条」，每来一条新结果，边界就往后挪一格，
    **每次请求都改历史** —— 缓存永远命中不了，省下的 token 还不够赔写缓存的钱。
    所以：超过 trigger 才动手，一动手就把能清的全清掉，而且省得不够
    clear_at_least 就干脆不动。清完以后远低于阈值，之后的请求都是纯追加，
    缓存又能命中，直到下一次涨过阈值。

    ── 锚点会失效 ────────────────────────────────────────────────────
    usage 锚点的前提是「它之前的消息没被改过」（见 Message.usage）。
    清掉第 k 条之后，第 k 条后面、清理**之前**就已存在的锚点量的是清理前的
    视图，都作废；清理**之后**才追加的锚点，量的就是清理后的视图，照样有效。
    _cleared 里记的「清理时历史有多长」就是用来区分这两种的。
    """

    # 占位的开头。测试、日志靠它认出「这是被清理过的结果」。
    CLEARED_PREFIX = "[这条工具结果已被清理，以节省上下文。"

    @classmethod
    def placeholder(cls, m: Message) -> str:
        """被清理的结果换成什么。

        不只是说「清掉了」，还要留下**线索**：原来是几行几列、哪些列
        （工具自己给的 summary）。模型看到「1 行：total=4242」就不必重查；
        看到「42 行 × 4 列（region, gmv, …）」能判断跟当前问题有没有关系。
        工具没给摘要时，至少告诉它原来有多大。

        同一条消息每次生成的占位必须一模一样，否则每次请求前缀都变，缓存全废。
        所以这里只依赖消息本身，不掺时间、计数之类会变的东西。
        """
        clue = m.summary or f"约 {len(m.content)} 字符"
        return (
            f"{cls.CLEARED_PREFIX}原结果：{clue}。"
            "调用参数还在上面的工具调用里；如果还需要完整数据，重新调用一次即可。]"
        )

    def __init__(
        self,
        trigger_tokens: int = 100_000,
        keep_recent: int = 3,
        clear_at_least: int = 10_000,
        exclude_tools: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self.trigger_tokens = trigger_tokens
        self.keep_recent = keep_recent
        self.clear_at_least = clear_at_least
        self.exclude_tools = frozenset(exclude_tools)
        # tool_call_id → 清理那一刻历史的长度。
        # 下标小于这个长度的消息都是清理之前加进来的。
        self._cleared: dict[str, int] = {}

    # ------------------------------------------------------------ 视图
    def render(self) -> list[Message]:
        out: list[Message] = []
        # 到目前为止，前面被清理的消息里「清理时历史长度」的最大值。
        # 下标比它小的锚点是在那次清理之前量的，已经不准了。
        stale_before = 0
        for i, m in enumerate(self._history):
            if m.role == "tool" and m.tool_call_id in self._cleared:
                stale_before = max(stale_before, self._cleared[m.tool_call_id])
                m = Message.tool_result(m.tool_call_id, self.placeholder(m))
            elif m.usage is not None and i < stale_before:
                # 生成新对象，不改原消息 —— 原消息还在历史和快照里
                m = replace(m, usage=None)
            out.append(m)
        return out

    # ------------------------------------------------------------ 清理
    def maintain(self, measure: Measure) -> ContextCleared | None:
        before = measure(self.render())
        if before <= self.trigger_tokens:
            return None

        targets = self._clearable()
        freed = sum(estimate_message(m) - estimate_text(self.placeholder(m)) for m in targets)
        if freed < self.clear_at_least:
            # 省得太少，不值得为此让缓存失效一次
            return None

        for m in targets:
            self._cleared[m.tool_call_id] = len(self._history)
        return ContextCleared(len(targets), before, measure(self.render()))

    def _clearable(self) -> list[Message]:
        """能清的工具结果：还没清过、不在排除名单里、不是最近 keep_recent 条。"""
        tool_names = {
            c.id: c.name for m in self._history for c in m.tool_calls
        }
        results = [m for m in self._history if m.role == "tool"]
        older = results[: max(len(results) - self.keep_recent, 0)]
        return [
            m for m in older
            if m.tool_call_id not in self._cleared
            and tool_names.get(m.tool_call_id) not in self.exclude_tools
        ]

    # ------------------------------------------------------ 状态 / 事务
    def clear(self) -> None:
        super().clear()
        self._cleared.clear()

    # 清理状态也是上下文的一部分，必须一起进快照。否则一轮失败回滚后，
    # 历史回去了，_cleared 里却还留着这一轮清理的记录。
    def snapshot(self) -> tuple[list[Message], dict[str, int]]:
        return list(self._history), dict(self._cleared)

    def restore(self, snapshot: tuple[list[Message], dict[str, int]]) -> None:
        history, cleared = snapshot
        self._history[:] = history
        self._cleared = dict(cleared)

    @property
    def cleared_count(self) -> int:
        return len(self._cleared)


class TurnWindowContext(FullContext):
    """按「回合」裁剪：只保留最近 max_turns 轮用户提问及其后续。

    回合的定义：从一条 role="user" 的消息开始，到下一条 role="user" 之前为止。
    以回合为单位切，就不会把 tool_calls / tool_result 拆散。
    """

    def __init__(self, max_turns: int = 6) -> None:
        super().__init__()
        self.max_turns = max_turns

    def render(self) -> list[Message]:
        history = self._history
        turn_starts = [i for i, m in enumerate(history) if m.role == "user"]
        if len(turn_starts) <= self.max_turns:
            return list(history)
        cut = turn_starts[-self.max_turns]
        return list(history[cut:])
