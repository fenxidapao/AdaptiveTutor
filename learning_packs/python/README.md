# Python 学习包（FeynmanTutor P0）

## 内容
- **knowledge-graph.json**：12 个知识点（ch1 变量/数据类型 6 个 + ch2 列表/字典 6 个），含前置依赖
- **exercises.jsonl**：40 道练习题（output 输出比对 / code 断言 / mcq 选择，全部规则判题，不调 LLM）
- **pretest.jsonl**：前测 10 题（mcq，覆盖 10 个知识点）
- **posttest.jsonl**：后测 10 题（**与 pretest 不同题、同知识点、同难度**，防止"记住答案"失真）
- **notes/**：12 个知识点的精编兜底讲解（每点 2-3 句，CourseRAG 检索失败时使用）

## 数据来源与许可（合规声明）
- 练习题/前后测：**原创**（基于 Python 教学常见题型设计，LLM 生成初稿 + 人工复核），无版权问题
- notes 讲解：**原创精编**（依据 Python 基础语法常识撰写）
- 讲解材料检索源：CourseRAG 的 `Python-廖雪峰教程-核心章节.md`
  - 许可：**CC BY-NC-SA 4.0**（已核实，见 CourseRAG 数据来源记录）
  - 使用方式：经 CourseRAG `/retrieve` HTTP 接口检索，**仅作讲解知识底座**，本项目不复制原文
- 本项目为非商用个人学习项目，符合 CC BY-NC 类许可使用范围

## 检索策略（2026-08-22 定稿，改动说明）
原方案为"调用侧 source 白名单过滤"，实测 CourseRAG `/retrieve`
接口**不支持 sources 参数**（`api_schemas.py` 仅 question/mode 两个字段，传参被静默忽略）。
改为：**检索 top-N 全量返回 → LLM 按内容相关性过滤 → notes 兜底**。
- 优点：零侵入 CourseRAG（服务边界清晰），不依赖 source 文件名（文件名可变）
- 成本：多一次 LLM 过滤调用，P0 规模下可忽略

## 对照实验说明
- ch1（变量/数据类型）与 ch2（列表/字典）：两章难度接近（同等级基础知识点），前后测提升差即教学法增量价值
- 教学法分组自 C2 起改为**注册时按 `group_name` 随机均衡分配**（不再按章节固定 lecture/feynman），
  以保证组间可比；章节与教学法不再绑定。设计见 `docs/EXPERIMENT.md`
