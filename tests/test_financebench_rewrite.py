"""rag_rewrite 的查询改写：模型回的 JSON 怎么解析、按公司年份类型挑文档（不调模型）。"""

from types import SimpleNamespace

from data_agent.core.messages import Usage
from evals.financebench.data import Question
from evals.financebench.e2e import filter_docs, plan_query

DOCS = {
    "3M_2018_10K": {"meta": {"company": "3M", "period": 2018, "doc_type": "10k"}},
    "3M_2023Q2_10Q": {"meta": {"company": "3M", "period": 2023, "doc_type": "10q"}},
    "COCACOLA_2017_10K": {"meta": {"company": "Coca-Cola", "period": 2017, "doc_type": "10k"}},
}
LIBRARY = SimpleNamespace(index=lambda: SimpleNamespace(docs=DOCS))
Q = Question("q1", "3M_2018_10K", "What is 3M's FY2018 capex?", "$1577.00", "metrics-generated", "", ())


def _llm(text: str):
    seen = {}

    def chat(messages, system):
        seen["system"] = system
        return SimpleNamespace(text=text, usage=Usage(input=10, output=5))
    return SimpleNamespace(chat=chat, seen=seen)


def test_解析模型回的JSON_代码块也认_公司列表进提示词():
    llm = _llm('```json\n{"company": "3M", "period": 2018, "doc_type": "10k", "queries": ["cash flows capex"]}\n```')
    plan, usage = plan_query(Q, llm, LIBRARY)
    assert plan["queries"] == ["cash flows capex"] and plan["company"] == "3M" and usage.input == 10
    assert "3M, Coca-Cola" in llm.seen["system"]


def test_改写失败就用原问题():
    plan, _ = plan_query(Q, _llm("sorry, I cannot"), LIBRARY)
    assert plan["queries"] == [Q.question]


def test_按公司年份类型挑文档_类型对不上放宽_没公司不过滤():
    assert filter_docs({"company": "3M", "period": 2018, "doc_type": "10k"}, LIBRARY) == ["3M_2018_10K"]
    assert filter_docs({"company": "3m", "period": "2018", "doc_type": "10q"}, LIBRARY) == ["3M_2018_10K"], \
        "类型猜错了放宽类型；大小写、年份写成字符串也认"
    assert filter_docs({"company": "3M", "period": None}, LIBRARY) == ["3M_2018_10K", "3M_2023Q2_10Q"]
    assert filter_docs({"company": None, "period": 2018}, LIBRARY) is None
    assert filter_docs({"company": "Tesla", "period": 2018}, LIBRARY) is None
